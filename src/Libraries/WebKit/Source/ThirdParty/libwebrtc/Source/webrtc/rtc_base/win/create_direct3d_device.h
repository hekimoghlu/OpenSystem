/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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
#ifndef RTC_BASE_WIN_CREATE_DIRECT3D_DEVICE_H_
#define RTC_BASE_WIN_CREATE_DIRECT3D_DEVICE_H_

#include <windows.graphics.directX.direct3d11.h>
#include <windows.graphics.directX.direct3d11.interop.h>
#include <winerror.h>
#include <wrl/client.h>

namespace webrtc {

// Callers must check the return value of ResolveCoreWinRTDirect3DDelayload()
// before using CreateDirect3DDeviceFromDXGIDevice().
bool ResolveCoreWinRTDirect3DDelayload();

// Allows for the creating of Direct3D Devices from a DXGI device on versions
// of Windows greater than Win7.
HRESULT CreateDirect3DDeviceFromDXGIDevice(
    IDXGIDevice* dxgi_device,
    ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice**
        out_d3d11_device);

}  // namespace webrtc

#endif  // RTC_BASE_WIN_CREATE_DIRECT3D_DEVICE_H_
