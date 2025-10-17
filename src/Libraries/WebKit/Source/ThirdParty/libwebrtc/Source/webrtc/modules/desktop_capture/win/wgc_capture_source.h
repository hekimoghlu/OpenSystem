/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_WGC_CAPTURE_SOURCE_H_
#define MODULES_DESKTOP_CAPTURE_WIN_WGC_CAPTURE_SOURCE_H_

#include <windows.graphics.capture.h>
#include <windows.graphics.h>
#include <wrl/client.h>

#include <memory>
#include <optional>

#include "modules/desktop_capture/desktop_capturer.h"
#include "modules/desktop_capture/desktop_geometry.h"

namespace webrtc {

// Abstract class to represent the source that WGC-based capturers capture
// from. Could represent an application window or a screen. Consumers should use
// the appropriate Wgc*SourceFactory class to create WgcCaptureSource objects
// of the appropriate type.
class WgcCaptureSource {
 public:
  explicit WgcCaptureSource(DesktopCapturer::SourceId source_id);
  virtual ~WgcCaptureSource();

  virtual DesktopVector GetTopLeft() = 0;
  // Lightweight version of IsCapturable which avoids allocating/deallocating
  // COM objects for each call. As such may return a different value than
  // IsCapturable.
  virtual bool ShouldBeCapturable();
  virtual bool IsCapturable();
  virtual bool FocusOnSource();
  virtual ABI::Windows::Graphics::SizeInt32 GetSize();
  HRESULT GetCaptureItem(
      Microsoft::WRL::ComPtr<
          ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>* result);
  DesktopCapturer::SourceId GetSourceId() { return source_id_; }

 protected:
  virtual HRESULT CreateCaptureItem(
      Microsoft::WRL::ComPtr<
          ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>* result) = 0;

 private:
  Microsoft::WRL::ComPtr<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>
      item_;
  const DesktopCapturer::SourceId source_id_;
};

class WgcCaptureSourceFactory {
 public:
  virtual ~WgcCaptureSourceFactory();

  virtual std::unique_ptr<WgcCaptureSource> CreateCaptureSource(
      DesktopCapturer::SourceId) = 0;
};

class WgcWindowSourceFactory final : public WgcCaptureSourceFactory {
 public:
  WgcWindowSourceFactory();

  // Disallow copy and assign.
  WgcWindowSourceFactory(const WgcWindowSourceFactory&) = delete;
  WgcWindowSourceFactory& operator=(const WgcWindowSourceFactory&) = delete;

  ~WgcWindowSourceFactory() override;

  std::unique_ptr<WgcCaptureSource> CreateCaptureSource(
      DesktopCapturer::SourceId) override;
};

class WgcScreenSourceFactory final : public WgcCaptureSourceFactory {
 public:
  WgcScreenSourceFactory();

  WgcScreenSourceFactory(const WgcScreenSourceFactory&) = delete;
  WgcScreenSourceFactory& operator=(const WgcScreenSourceFactory&) = delete;

  ~WgcScreenSourceFactory() override;

  std::unique_ptr<WgcCaptureSource> CreateCaptureSource(
      DesktopCapturer::SourceId) override;
};

// Class for capturing application windows.
class WgcWindowSource final : public WgcCaptureSource {
 public:
  explicit WgcWindowSource(DesktopCapturer::SourceId source_id);

  WgcWindowSource(const WgcWindowSource&) = delete;
  WgcWindowSource& operator=(const WgcWindowSource&) = delete;

  ~WgcWindowSource() override;

  DesktopVector GetTopLeft() override;
  ABI::Windows::Graphics::SizeInt32 GetSize() override;
  bool ShouldBeCapturable() override;
  bool IsCapturable() override;
  bool FocusOnSource() override;

 private:
  HRESULT CreateCaptureItem(
      Microsoft::WRL::ComPtr<
          ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>* result)
      override;
};

// Class for capturing screens/monitors/displays.
class WgcScreenSource final : public WgcCaptureSource {
 public:
  explicit WgcScreenSource(DesktopCapturer::SourceId source_id);

  WgcScreenSource(const WgcScreenSource&) = delete;
  WgcScreenSource& operator=(const WgcScreenSource&) = delete;

  ~WgcScreenSource() override;

  DesktopVector GetTopLeft() override;
  ABI::Windows::Graphics::SizeInt32 GetSize() override;
  bool IsCapturable() override;

 private:
  HRESULT CreateCaptureItem(
      Microsoft::WRL::ComPtr<
          ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>* result)
      override;

  // To maintain compatibility with other capturers, this class accepts a
  // device index as it's SourceId. However, WGC requires we use an HMONITOR to
  // describe which screen to capture. So, we internally convert the supplied
  // device index into an HMONITOR when `IsCapturable()` is called.
  std::optional<HMONITOR> hmonitor_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_WGC_CAPTURE_SOURCE_H_
