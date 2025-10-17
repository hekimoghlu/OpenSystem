/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_WIN_D3D_DEVICE_H_
#define MODULES_DESKTOP_CAPTURE_WIN_D3D_DEVICE_H_

#include <comdef.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <vector>

namespace webrtc {

// A wrapper of ID3D11Device and its corresponding context and IDXGIAdapter.
// This class represents one video card in the system.
class D3dDevice {
 public:
  D3dDevice(const D3dDevice& other);
  D3dDevice(D3dDevice&& other);
  ~D3dDevice();

  ID3D11Device* d3d_device() const { return d3d_device_.Get(); }

  ID3D11DeviceContext* context() const { return context_.Get(); }

  IDXGIDevice* dxgi_device() const { return dxgi_device_.Get(); }

  IDXGIAdapter* dxgi_adapter() const { return dxgi_adapter_.Get(); }

  // Returns all D3dDevice instances on the system. Returns an empty vector if
  // anything wrong.
  static std::vector<D3dDevice> EnumDevices();

 private:
  // Instances of D3dDevice should only be created by EnumDevices() static
  // function.
  D3dDevice();

  // Initializes the D3dDevice from an IDXGIAdapter.
  bool Initialize(const Microsoft::WRL::ComPtr<IDXGIAdapter>& adapter);

  Microsoft::WRL::ComPtr<ID3D11Device> d3d_device_;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
  Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device_;
  Microsoft::WRL::ComPtr<IDXGIAdapter> dxgi_adapter_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_WIN_D3D_DEVICE_H_
