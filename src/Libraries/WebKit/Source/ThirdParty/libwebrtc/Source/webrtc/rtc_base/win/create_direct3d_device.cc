/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include "rtc_base/win/create_direct3d_device.h"

#include <libloaderapi.h>

#include <utility>

namespace {

FARPROC LoadD3D11Function(const char* function_name) {
  static HMODULE const handle =
      ::LoadLibraryExW(L"d3d11.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  return handle ? ::GetProcAddress(handle, function_name) : nullptr;
}

decltype(&::CreateDirect3D11DeviceFromDXGIDevice)
GetCreateDirect3D11DeviceFromDXGIDevice() {
  static decltype(&::CreateDirect3D11DeviceFromDXGIDevice) const function =
      reinterpret_cast<decltype(&::CreateDirect3D11DeviceFromDXGIDevice)>(
          LoadD3D11Function("CreateDirect3D11DeviceFromDXGIDevice"));
  return function;
}

}  // namespace

namespace webrtc {

bool ResolveCoreWinRTDirect3DDelayload() {
  return GetCreateDirect3D11DeviceFromDXGIDevice();
}

HRESULT CreateDirect3DDeviceFromDXGIDevice(
    IDXGIDevice* dxgi_device,
    ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice**
        out_d3d11_device) {
  decltype(&::CreateDirect3D11DeviceFromDXGIDevice) create_d3d11_device_func =
      GetCreateDirect3D11DeviceFromDXGIDevice();
  if (!create_d3d11_device_func)
    return E_FAIL;

  Microsoft::WRL::ComPtr<IInspectable> inspectableSurface;
  HRESULT hr = create_d3d11_device_func(dxgi_device, &inspectableSurface);
  if (FAILED(hr))
    return hr;

  return inspectableSurface->QueryInterface(IID_PPV_ARGS(out_d3d11_device));
}

}  // namespace webrtc
