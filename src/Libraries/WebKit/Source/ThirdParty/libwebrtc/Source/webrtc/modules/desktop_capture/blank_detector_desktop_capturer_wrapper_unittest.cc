/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "modules/desktop_capture/blank_detector_desktop_capturer_wrapper.h"

#include <memory>
#include <utility>

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_frame_generator.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/desktop_region.h"
#include "modules/desktop_capture/fake_desktop_capturer.h"
#include "test/gtest.h"

namespace webrtc {

class BlankDetectorDesktopCapturerWrapperTest
    : public ::testing::Test,
      public DesktopCapturer::Callback {
 public:
  BlankDetectorDesktopCapturerWrapperTest();
  ~BlankDetectorDesktopCapturerWrapperTest() override;

 protected:
  void PerfTest(DesktopCapturer* capturer);

  const int frame_width_ = 1024;
  const int frame_height_ = 768;
  std::unique_ptr<BlankDetectorDesktopCapturerWrapper> wrapper_;
  DesktopCapturer* capturer_ = nullptr;
  BlackWhiteDesktopFramePainter painter_;
  int num_frames_captured_ = 0;
  DesktopCapturer::Result last_result_ = DesktopCapturer::Result::SUCCESS;
  std::unique_ptr<DesktopFrame> last_frame_;

 private:
  // DesktopCapturer::Callback interface.
  void OnCaptureResult(DesktopCapturer::Result result,
                       std::unique_ptr<DesktopFrame> frame) override;

  PainterDesktopFrameGenerator frame_generator_;
};

BlankDetectorDesktopCapturerWrapperTest::
    BlankDetectorDesktopCapturerWrapperTest() {
  frame_generator_.size()->set(frame_width_, frame_height_);
  frame_generator_.set_desktop_frame_painter(&painter_);
  std::unique_ptr<DesktopCapturer> capturer(new FakeDesktopCapturer());
  FakeDesktopCapturer* fake_capturer =
      static_cast<FakeDesktopCapturer*>(capturer.get());
  fake_capturer->set_frame_generator(&frame_generator_);
  capturer_ = fake_capturer;
  wrapper_.reset(new BlankDetectorDesktopCapturerWrapper(
      std::move(capturer), RgbaColor(0, 0, 0, 0)));
  wrapper_->Start(this);
}

BlankDetectorDesktopCapturerWrapperTest::
    ~BlankDetectorDesktopCapturerWrapperTest() = default;

void BlankDetectorDesktopCapturerWrapperTest::OnCaptureResult(
    DesktopCapturer::Result result,
    std::unique_ptr<DesktopFrame> frame) {
  last_result_ = result;
  last_frame_ = std::move(frame);
  num_frames_captured_++;
}

void BlankDetectorDesktopCapturerWrapperTest::PerfTest(
    DesktopCapturer* capturer) {
  for (int i = 0; i < 10000; i++) {
    capturer->CaptureFrame();
    ASSERT_EQ(num_frames_captured_, i + 1);
  }
}

TEST_F(BlankDetectorDesktopCapturerWrapperTest, ShouldDetectBlankFrame) {
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 1);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::ERROR_TEMPORARY);
  ASSERT_FALSE(last_frame_);
}

TEST_F(BlankDetectorDesktopCapturerWrapperTest, ShouldPassBlankDetection) {
  painter_.updated_region()->AddRect(DesktopRect::MakeXYWH(0, 0, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 1);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);

  painter_.updated_region()->AddRect(
      DesktopRect::MakeXYWH(frame_width_ - 100, frame_height_ - 100, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 2);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);

  painter_.updated_region()->AddRect(
      DesktopRect::MakeXYWH(0, frame_height_ - 100, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 3);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);

  painter_.updated_region()->AddRect(
      DesktopRect::MakeXYWH(frame_width_ - 100, 0, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 4);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);

  painter_.updated_region()->AddRect(DesktopRect::MakeXYWH(
      (frame_width_ >> 1) - 50, (frame_height_ >> 1) - 50, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 5);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);
}

TEST_F(BlankDetectorDesktopCapturerWrapperTest,
       ShouldNotCheckAfterANonBlankFrameReceived) {
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 1);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::ERROR_TEMPORARY);
  ASSERT_FALSE(last_frame_);

  painter_.updated_region()->AddRect(
      DesktopRect::MakeXYWH(frame_width_ - 100, 0, 100, 100));
  wrapper_->CaptureFrame();
  ASSERT_EQ(num_frames_captured_, 2);
  ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
  ASSERT_TRUE(last_frame_);

  for (int i = 0; i < 100; i++) {
    wrapper_->CaptureFrame();
    ASSERT_EQ(num_frames_captured_, i + 3);
    ASSERT_EQ(last_result_, DesktopCapturer::Result::SUCCESS);
    ASSERT_TRUE(last_frame_);
  }
}

// There is no perceptible impact by using BlankDetectorDesktopCapturerWrapper.
// i.e. less than 0.2ms per frame.
// [ OK ] DISABLED_Performance (10210 ms)
// [ OK ] DISABLED_PerformanceComparison (8791 ms)
TEST_F(BlankDetectorDesktopCapturerWrapperTest, DISABLED_Performance) {
  PerfTest(wrapper_.get());
}

TEST_F(BlankDetectorDesktopCapturerWrapperTest,
       DISABLED_PerformanceComparison) {
  capturer_->Start(this);
  PerfTest(capturer_);
}

}  // namespace webrtc
