/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#ifndef MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_HELP_FUNCTIONS_DS_H_
#define MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_HELP_FUNCTIONS_DS_H_

#include <dshow.h>

#include <type_traits>
#include <utility>

#include "api/scoped_refptr.h"
#include "rtc_base/ref_counter.h"

DEFINE_GUID(MEDIASUBTYPE_I420,
            0x30323449,
            0x0000,
            0x0010,
            0x80,
            0x00,
            0x00,
            0xAA,
            0x00,
            0x38,
            0x9B,
            0x71);
DEFINE_GUID(MEDIASUBTYPE_HDYC,
            0x43594448,
            0x0000,
            0x0010,
            0x80,
            0x00,
            0x00,
            0xAA,
            0x00,
            0x38,
            0x9B,
            0x71);

#define RELEASE_AND_CLEAR(p) \
  if (p) {                   \
    (p)->Release();          \
    (p) = NULL;              \
  }

namespace webrtc {
namespace videocapturemodule {
LONGLONG GetMaxOfFrameArray(LONGLONG* maxFps, long size);

IPin* GetInputPin(IBaseFilter* filter);
IPin* GetOutputPin(IBaseFilter* filter, REFGUID Category);
BOOL PinMatchesCategory(IPin* pPin, REFGUID Category);
void ResetMediaType(AM_MEDIA_TYPE* media_type);
void FreeMediaType(AM_MEDIA_TYPE* media_type);
HRESULT CopyMediaType(AM_MEDIA_TYPE* target, const AM_MEDIA_TYPE* source);

// Helper function to make using scoped_refptr with COM interface pointers
// a little less awkward. rtc::scoped_refptr doesn't support the & operator
// or a way to receive values via an out ptr.
// The function is intentionally not called QueryInterface to make things less
// confusing for the compiler to figure out what the caller wants to do when
// called from within the context of a class that also implements COM
// interfaces.
template <class T>
HRESULT GetComInterface(IUnknown* object, rtc::scoped_refptr<T>* ptr) {
  // This helper function is not meant to magically free ptr. If we do that
  // we add code bloat to most places where it's not needed and make the code
  // less readable since it's not clear at the call site that the pointer
  // would get freed even inf QI() fails.
  RTC_DCHECK(!ptr->get());
  void* new_ptr = nullptr;
  HRESULT hr = object->QueryInterface(__uuidof(T), &new_ptr);
  if (SUCCEEDED(hr))
    ptr->swap(reinterpret_cast<T**>(&new_ptr));
  return hr;
}

// Provides a reference count implementation for COM (IUnknown derived) classes.
// The implementation uses atomics for managing the ref count.
template <class T>
class ComRefCount : public T {
 public:
  ComRefCount() {}

  template <class P0>
  explicit ComRefCount(P0&& p0) : T(std::forward<P0>(p0)) {}

  STDMETHOD_(ULONG, AddRef)() override {
    ref_count_.IncRef();
    return 1;
  }

  STDMETHOD_(ULONG, Release)() override {
    const auto status = ref_count_.DecRef();
    if (status == RefCountReleaseStatus::kDroppedLastRef) {
      delete this;
      return 0;
    }
    return 1;
  }

 protected:
  ~ComRefCount() {}

 private:
  webrtc::webrtc_impl::RefCounter ref_count_{0};
};

}  // namespace videocapturemodule
}  // namespace webrtc
#endif  // MODULES_VIDEO_CAPTURE_MAIN_SOURCE_WINDOWS_HELP_FUNCTIONS_DS_H_
