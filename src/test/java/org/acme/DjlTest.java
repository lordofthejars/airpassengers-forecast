package org.acme;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

public class DjlTest {

    @Test
    public void testNdArray() {

        try (Model model = Model.newInstance("deepar")) {
           NDManager manager = model.getNDManager();
           NDArray array = manager.arange(8f);

           NDList split = array.split(new long[] {5});
           System.out.println("*****");
           System.out.println(Arrays.toString(split.get(0).toFloatArray()));
           System.out.println(Arrays.toString(split.get(1).toFloatArray()));
        }

    }

}
