/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include <initguid.h>  // Must come before the help_functions_ds.h include so
                       // that DEFINE_GUID() entries will be defined in this
                       // object file.

#include <cguid.h>

#include "modules/video_capture/windows/help_functions_ds.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace videocapturemodule {
// This returns minimum :), which will give max frame rate...
LONGLONG GetMaxOfFrameArray(LONGLONG* maxFps, long size) {
  if (!maxFps || size <= 0) {
    return 0;
  }
  LONGLONG maxFPS = maxFps[0];
  for (int i = 0; i < size; i++) {
    if (maxFPS > maxFps[i])
      maxFPS = maxFps[i];
  }
  return maxFPS;
}

IPin* GetInputPin(IBaseFilter* filter) {
  IPin* pin = NULL;
  IEnumPins* pPinEnum = NULL;
  filter->EnumPins(&pPinEnum);
  if (pPinEnum == NULL) {
    return NULL;
  }

  // get first unconnected pin
  pPinEnum->Reset();  // set to first pin

  while (S_OK == pPinEnum->Next(1, &pin, NULL)) {
    PIN_DIRECTION pPinDir;
    pin->QueryDirection(&pPinDir);
    if (PINDIR_INPUT == pPinDir)  // This is an input pin
    {
      IPin* tempPin = NULL;
      if (S_OK != pin->ConnectedTo(&tempPin))  // The pint is not connected
      {
        pPinEnum->Release();
        return pin;
      }
    }
    pin->Release();
  }
  pPinEnum->Release();
  return NULL;
}

IPin* GetOutputPin(IBaseFilter* filter, REFGUID Category) {
  IPin* pin = NULL;
  IEnumPins* pPinEnum = NULL;
  filter->EnumPins(&pPinEnum);
  if (pPinEnum == NULL) {
    return NULL;
  }
  // get first unconnected pin
  pPinEnum->Reset();  // set to first pin
  while (S_OK == pPinEnum->Next(1, &pin, NULL)) {
    PIN_DIRECTION pPinDir;
    pin->QueryDirection(&pPinDir);
    if (PINDIR_OUTPUT == pPinDir)  // This is an output pin
    {
      if (Category == GUID_NULL || PinMatchesCategory(pin, Category)) {
        pPinEnum->Release();
        return pin;
      }
    }
    pin->Release();
    pin = NULL;
  }
  pPinEnum->Release();
  return NULL;
}

BOOL PinMatchesCategory(IPin* pPin, REFGUID Category) {
  BOOL bFound = FALSE;
  IKsPropertySet* pKs = NULL;
  HRESULT hr = pPin->QueryInterface(IID_PPV_ARGS(&pKs));
  if (SUCCEEDED(hr)) {
    GUID PinCategory;
    DWORD cbReturned;
    hr = pKs->Get(AMPROPSETID_Pin, AMPROPERTY_PIN_CATEGORY, NULL, 0,
                  &PinCategory, sizeof(GUID), &cbReturned);
    if (SUCCEEDED(hr) && (cbReturned == sizeof(GUID))) {
      bFound = (PinCategory == Category);
    }
    pKs->Release();
  }
  return bFound;
}

void ResetMediaType(AM_MEDIA_TYPE* media_type) {
  if (!media_type)
    return;
  if (media_type->cbFormat != 0) {
    CoTaskMemFree(media_type->pbFormat);
    media_type->cbFormat = 0;
    media_type->pbFormat = nullptr;
  }
  if (media_type->pUnk) {
    media_type->pUnk->Release();
    media_type->pUnk = nullptr;
  }
}

void FreeMediaType(AM_MEDIA_TYPE* media_type) {
  if (!media_type)
    return;
  ResetMediaType(media_type);
  CoTaskMemFree(media_type);
}

HRESULT CopyMediaType(AM_MEDIA_TYPE* target, const AM_MEDIA_TYPE* source) {
  RTC_DCHECK_NE(source, target);
  *target = *source;
  if (source->cbFormat != 0) {
    RTC_DCHECK(source->pbFormat);
    target->pbFormat =
        reinterpret_cast<BYTE*>(CoTaskMemAlloc(source->cbFormat));
    if (target->pbFormat == nullptr) {
      target->cbFormat = 0;
      return E_OUTOFMEMORY;
    } else {
      CopyMemory(target->pbFormat, source->pbFormat, target->cbFormat);
    }
  }

  if (target->pUnk != nullptr)
    target->pUnk->AddRef();

  return S_OK;
}

wchar_t* DuplicateWideString(const wchar_t* str) {
  size_t len = lstrlenW(str);
  wchar_t* ret =
      reinterpret_cast<LPWSTR>(CoTaskMemAlloc((len + 1) * sizeof(wchar_t)));
  lstrcpyW(ret, str);
  return ret;
}

}  // namespace videocapturemodule
}  // namespace webrtc
