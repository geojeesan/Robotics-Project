#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/motor.h>
#include <arm.h>
#include <gripper.h>
#include <base.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TIME_STEP 32

static void step() {
  if (wb_robot_step(TIME_STEP) == -1) {
    wb_robot_cleanup();
    exit(0);
  }
}

int main() {
  wb_robot_init();

  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, TIME_STEP);

  arm_init();
  gripper_init();
  base_init();

  printf("Vision-based Trash Detection Active...\n");

  while (wb_robot_step(TIME_STEP) != -1) {
    const unsigned char *image = wb_camera_get_image(camera);
    int width = wb_camera_get_width(camera);
    int height = wb_camera_get_height(camera);

    int red_x_sum = 0, red_y_sum = 0, red_count = 0;

    // Simple red-trash detection
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int r = wb_camera_image_get_red(image, width, x, y);
        int g = wb_camera_image_get_green(image, width, x, y);
        int b = wb_camera_image_get_blue(image, width, x, y);

        if (r > 120 && g < 80 && b < 80) {
          red_x_sum += x;
          red_y_sum += y;
          red_count++;
        }
      }
    }

    if (red_count > 100) {  // object detected
      int red_x_avg = red_x_sum / red_count;
      int red_y_avg = red_y_sum / red_count;

      printf("Red trash detected at pixel: (%d, %d)\n", red_x_avg, red_y_avg);

      // Move forward if object is centered
      if (abs(red_x_avg - width / 2) < 30) {
        base_forwards();
      } else if (red_x_avg < width / 2 - 30) {
        base_turn_left();
      } else {
        base_turn_right();
      }

      step();

      // Simulate pick-up sequence
      base_reset();
      arm_set_height(ARM_FRONT_FLOOR);
      gripper_release();
      step();
      gripper_grip();
      printf("Picked up trash!\n");
      arm_set_height(ARM_BACK_PLATE_LOW);
      step();
    }
  }

  wb_robot_cleanup();
  return 0;
}
