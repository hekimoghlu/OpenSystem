/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
/* 	Copyright (c) 1992 NeXT Computer, Inc.  All rights reserved. 
 *
 * IOHIDevice.h - Common Event Source object class.
 *
 * HISTORY
 * 22 May 1992    Mike Paquette at NeXT
 *      Created. 
 * 4  Aug 1993	  Erik Kay at NeXT
 *	API cleanup
 * 5  Aug 1993	  Erik Kay at NeXT
 *	added ivar space for future expansion
 */

#ifndef _IOHIDEVICE_H
#define _IOHIDEVICE_H

#include <IOKit/IOService.h>
#include <IOKit/IOLocks.h>

typedef enum {
  kHIUnknownDevice          = 0,
  kHIKeyboardDevice         = 1,
  kHIRelativePointingDevice = 2
} IOHIDKind;

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIDevice : public IOService
#else
class IOHIDevice : public IOService
#endif
{
  OSDeclareDefaultStructors(IOHIDevice);

public:
  virtual bool init(OSDictionary * properties = 0) APPLE_KEXT_OVERRIDE;
  virtual void free(void) APPLE_KEXT_OVERRIDE;
  virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
  virtual bool open(  IOService *    forClient,
                      IOOptionBits   options = 0,
                      void *         arg = 0 ) APPLE_KEXT_OVERRIDE;

  virtual UInt32    deviceType();
  virtual IOHIDKind hidKind();
  virtual UInt32    interfaceID();
  virtual bool 	    updateProperties(void);
  virtual IOReturn  setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;
  virtual IOReturn  setParamProperties(OSDictionary * dict);
  virtual UInt64    getGUID();
  
  static SInt32		GenerateKey(OSObject *object);
};

#endif /* !_IOHIDEVICE_H */
