package com.example.stableDiffusionApp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    Button btnGenerate;
    ImageView modelImage;
    TextView modelWaiting;

    private static final String TAG = "MainActivity";

    Module model;

    int inputSize = 512;

    int width = 256;
    int height = 256;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnGenerate = findViewById(R.id.btnGenerate);
        modelImage = findViewById(R.id.modelImage);
        modelWaiting = findViewById(R.id.modelWaiting);

        try {
            model = LiteModuleLoader.load(assetFilePath());
        } catch (IOException e) {
            Log.e(TAG, "Unable to load model", e);
        }

        btnGenerate.setOnClickListener(view -> {
            btnGenerate.setClickable(false);
            modelImage.setVisibility(View.INVISIBLE);
            modelWaiting.setVisibility(View.VISIBLE);

            Tensor inputTensor = generateTensor(inputSize);

            new Thread(() -> {
                float[] outputArr = model.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();

                for (int i = 0; i < outputArr.length; i++) {
                    outputArr[i] = Math.min(Math.max(outputArr[i], 0), 255);
                }

                //This is printing all zeroes
                System.out.println(Arrays.toString(outputArr));
            }).start();

        });
    }

    private Tensor generateTensor(int size) {
        Random rand = new Random();
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = rand.nextGaussian();
        }

        long[] s = {1, size};
        return Tensor.fromBlob(arr, s);
    }

    private String assetFilePath() throws IOException {
        File file = new File(this.getFilesDir(), "stableDiffusion.pt");
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open("stableDiffusion.pt")) {
            try (OutputStream os = Files.newOutputStream(file.toPath())) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}