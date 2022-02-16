package com.example.application;
/*
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

public class SecondActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
    }
}
*/

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;

import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.widget.ImageView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;


public class SecondActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {
    Handler handler;
    private int sensorOrientation;
    String imageString = "";
    private ImageView imageView2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        handler = new Handler();
        imageView2 = (ImageView) findViewById(R.id.image_view2);

        //TODO ask for camera permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA}, 121);
        } else {
            //TODO show live camera footage
            setFragment();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //TODO show live camera footage
        if (grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            setFragment();
        } else {
            finish();
        }
    }

    //TODO fragment which show llive footage from camera
    int previewHeight = 0, previewWidth = 0;

    protected void setFragment() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[1];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

        Fragment fragment;
        CameraConnectionFragment camera2Fragment =
                CameraConnectionFragment.newInstance(
                        new CameraConnectionFragment.ConnectionCallback() {
                            @Override
                            public void onPreviewSizeChosen(final Size size, final int rotation) {
                                previewHeight = size.getHeight();
                                previewWidth = size.getWidth();
                                sensorOrientation = rotation - getScreenOrientation();
                            }
                        },
                        this,
                        R.layout.camera_fragment,
                        new Size(640, 480));

        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;

        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }


    //TODO getting frames of live camera footage and passing them to model
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    @Override
    public void onImageAvailable(ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };

            processImage();
            imageView2.setImageBitmap(rgbFrameBitmap);

        } catch (final Exception e) {
            Log.d("tryError",e.getMessage());
            return;
        }

    }
/*
    private void processImage() {
        imageConverter.run();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        //Do your work here
        postInferenceCallback.run();
    }


 */


    final Python py = Python.getInstance();
    private void processImage() {
        imageConverter.run();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        //Do your work here
        imageString = getStringImage(rgbFrameBitmap);
        Log.i("python", imageString);
        PyObject pyo = py.getModule("detect");
        PyObject obj = pyo.callAttr("det", imageString);
        String str = obj.toString();
        Log.i("python",str);
        byte data_bitmap[] = android.util.Base64.decode(str, Base64.DEFAULT);
        rgbFrameBitmap = BitmapFactory.decodeByteArray(data_bitmap, 0, data_bitmap.length);
        Log.i("python","test");

        //imageView2.setImageBitmap(rgbFrameBitmap);
        postInferenceCallback.run();
    }




    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos);
        byte[] b = baos.toByteArray();
        String temp = Base64.encodeToString(b, Base64.DEFAULT);
        return temp;
    }
}