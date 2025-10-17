/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
#include <IOKit/assert.h>
#include "ApplePS2MouseDevice.h"
#include "ApplePS2Controller.h"

// =============================================================================
// ApplePS2MouseDevice Class Implementation
//

#define super IOService
OSDefineMetaClassAndStructors(ApplePS2MouseDevice, IOService);

bool ApplePS2MouseDevice::attach(IOService * provider)
{
  if( !super::attach(provider) )  return false;

  assert(_controller == 0);
  _controller = (ApplePS2Controller *)provider;
  _controller->retain();

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::detach( IOService * provider )
{
  assert(_controller == provider);
  _controller->release();
  _controller = 0;

  super::detach(provider);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::installInterruptAction(OSObject *         target,
                                                 PS2InterruptAction action)
{
  _controller->installInterruptAction(kDT_Mouse, target, action);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::uninstallInterruptAction()
{
  _controller->uninstallInterruptAction(kDT_Mouse);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::installPowerControlAction(OSObject *            target,
                                                    PS2PowerControlAction action)
{
  _controller->installPowerControlAction(kDT_Mouse, target, action);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::uninstallPowerControlAction()
{
  _controller->uninstallPowerControlAction(kDT_Mouse);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

PS2Request * ApplePS2MouseDevice::allocateRequest()
{
  return _controller->allocateRequest();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::freeRequest(PS2Request * request)
{
  _controller->freeRequest(request);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

bool ApplePS2MouseDevice::submitRequest(PS2Request * request)
{
  return _controller->submitRequest(request);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void ApplePS2MouseDevice::submitRequestAndBlock(PS2Request * request)
{
  _controller->submitRequestAndBlock(request);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 0);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 1);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 2);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 3);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 4);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 5);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 6);
OSMetaClassDefineReservedUnused(ApplePS2MouseDevice, 7);
