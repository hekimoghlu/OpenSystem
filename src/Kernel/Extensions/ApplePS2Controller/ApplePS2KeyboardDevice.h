/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#ifndef _APPLEPS2KEYBOARDDEVICE_H
#define _APPLEPS2KEYBOARDDEVICE_H

#include "ApplePS2Device.h"

class ApplePS2Controller;

class ApplePS2KeyboardDevice : public IOService
{
  OSDeclareDefaultStructors(ApplePS2KeyboardDevice);

private:
  ApplePS2Controller * _controller;

protected:
  struct ExpansionData { /* */ };
  ExpansionData * _expansionData;

public:
  virtual bool attach(IOService * provider) override;
  virtual void detach(IOService * provider) override;

  // Interrupt Handling Routines

  virtual void installInterruptAction(OSObject *, PS2InterruptAction);
  virtual void uninstallInterruptAction();

  // Request Submission Routines

  virtual PS2Request * allocateRequest();
  virtual void         freeRequest(PS2Request * request);
  virtual bool         submitRequest(PS2Request * request);
  virtual void         submitRequestAndBlock(PS2Request * request);

  // Power Control Handling Routines

  virtual void installPowerControlAction(OSObject *, PS2PowerControlAction);
  virtual void uninstallPowerControlAction();

  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 0);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 1);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 2);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 3);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 4);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 5);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 6);
  OSMetaClassDeclareReservedUnused(ApplePS2KeyboardDevice, 7);
};

#endif /* !_APPLEPS2KEYBOARDDEVICE_H */
