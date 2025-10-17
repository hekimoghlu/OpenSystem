/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#include "modules/desktop_capture/win/wgc_capture_source.h"

#include <windows.graphics.capture.h>
#include <wrl/client.h>

#include <utility>

#include "modules/desktop_capture/desktop_capture_types.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "modules/desktop_capture/win/screen_capture_utils.h"
#include "modules/desktop_capture/win/test_support/test_window.h"
#include "modules/desktop_capture/win/wgc_capturer_win.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/win/scoped_com_initializer.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

const WCHAR kWindowTitle[] = L"WGC Capture Source Test Window";

const int kFirstXCoord = 25;
const int kFirstYCoord = 50;
const int kSecondXCoord = 50;
const int kSecondYCoord = 75;

}  // namespace

class WgcCaptureSourceTest : public ::testing::TestWithParam<CaptureType> {
 public:
  void SetUp() override {
    com_initializer_ =
        std::make_unique<ScopedCOMInitializer>(ScopedCOMInitializer::kMTA);
    ASSERT_TRUE(com_initializer_->Succeeded());
  }

  void TearDown() override {
    if (window_open_) {
      DestroyTestWindow(window_info_);
    }
  }

  void SetUpForWindowSource() {
    window_info_ = CreateTestWindow(kWindowTitle);
    window_open_ = true;
    source_id_ = reinterpret_cast<DesktopCapturer::SourceId>(window_info_.hwnd);
    source_factory_ = std::make_unique<WgcWindowSourceFactory>();
  }

  void SetUpForScreenSource() {
    source_id_ = kFullDesktopScreenId;
    source_factory_ = std::make_unique<WgcScreenSourceFactory>();
  }

 protected:
  std::unique_ptr<ScopedCOMInitializer> com_initializer_;
  std::unique_ptr<WgcCaptureSourceFactory> source_factory_;
  std::unique_ptr<WgcCaptureSource> source_;
  DesktopCapturer::SourceId source_id_;
  WindowInfo window_info_;
  bool window_open_ = false;
};

// Window specific test
TEST_F(WgcCaptureSourceTest, WindowPosition) {
  if (!IsWgcSupported(CaptureType::kWindow)) {
    RTC_LOG(LS_INFO)
        << "Skipping WgcCapturerWinTests on unsupported platforms.";
    GTEST_SKIP();
  }

  SetUpForWindowSource();
  source_ = source_factory_->CreateCaptureSource(source_id_);
  ASSERT_TRUE(source_);
  EXPECT_EQ(source_->GetSourceId(), source_id_);

  MoveTestWindow(window_info_.hwnd, kFirstXCoord, kFirstYCoord);
  DesktopVector source_vector = source_->GetTopLeft();
  EXPECT_EQ(source_vector.x(), kFirstXCoord);
  EXPECT_EQ(source_vector.y(), kFirstYCoord);

  MoveTestWindow(window_info_.hwnd, kSecondXCoord, kSecondYCoord);
  source_vector = source_->GetTopLeft();
  EXPECT_EQ(source_vector.x(), kSecondXCoord);
  EXPECT_EQ(source_vector.y(), kSecondYCoord);
}

// Screen specific test
TEST_F(WgcCaptureSourceTest, ScreenPosition) {
  if (!IsWgcSupported(CaptureType::kScreen)) {
    RTC_LOG(LS_INFO)
        << "Skipping WgcCapturerWinTests on unsupported platforms.";
    GTEST_SKIP();
  }

  SetUpForScreenSource();
  source_ = source_factory_->CreateCaptureSource(source_id_);
  ASSERT_TRUE(source_);
  EXPECT_EQ(source_id_, source_->GetSourceId());

  DesktopRect screen_rect = GetFullscreenRect();
  DesktopVector source_vector = source_->GetTopLeft();
  EXPECT_EQ(source_vector.x(), screen_rect.left());
  EXPECT_EQ(source_vector.y(), screen_rect.top());
}

// Source agnostic test
TEST_P(WgcCaptureSourceTest, CreateSource) {
  if (!IsWgcSupported(GetParam())) {
    RTC_LOG(LS_INFO)
        << "Skipping WgcCapturerWinTests on unsupported platforms.";
    GTEST_SKIP();
  }

  if (GetParam() == CaptureType::kWindow) {
    SetUpForWindowSource();
  } else {
    SetUpForScreenSource();
  }

  source_ = source_factory_->CreateCaptureSource(source_id_);
  ASSERT_TRUE(source_);
  EXPECT_EQ(source_id_, source_->GetSourceId());
  EXPECT_TRUE(source_->IsCapturable());

  Microsoft::WRL::ComPtr<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>
      item;
  EXPECT_TRUE(SUCCEEDED(source_->GetCaptureItem(&item)));
  EXPECT_TRUE(item);
}

INSTANTIATE_TEST_SUITE_P(SourceAgnostic,
                         WgcCaptureSourceTest,
                         ::testing::Values(CaptureType::kWindow,
                                           CaptureType::kScreen));

}  // namespace webrtc
