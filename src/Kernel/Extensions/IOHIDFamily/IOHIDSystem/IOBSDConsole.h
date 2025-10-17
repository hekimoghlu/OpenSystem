/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#ifndef _IOBSDCONSOLE_H
#define _IOBSDCONSOLE_H

#include <IOKit/IOService.h>

class IOHIKeyboard;

class IOBSDConsole : public IOService
{
  OSDeclareDefaultStructors(IOBSDConsole);

private:
    IOService * fAudioOut;

  static bool publishNotificationHandler(
			IOBSDConsole * self, void * ref,
			IOService * newService );

  virtual void arbitrateForKeyboard( IOHIKeyboard * kb );

public:

  static void keyboardEvent(OSObject * target,
     /* eventType */        unsigned   eventType,
     /* flags */            unsigned   flags,
     /* keyCode */          unsigned   key,
     /* charCode */         unsigned   charCode,
     /* charSet */          unsigned   charSet,
     /* originalCharCode */ unsigned   origCharCode,
     /* originalCharSet */  unsigned   origCharSet,
     /* keyboardType */     unsigned   keyboardType,
     /* repeat */           bool       repeat,
     /* atTime */           AbsoluteTime ts,
                            OSObject * sender,
                            void *     refcon);

  static void updateEventFlags(
                            OSObject * target, 
                            unsigned flags,
                            OSObject * sender,
                            void *     refcon);

  virtual bool start(IOService * provider);

  virtual IOReturn message(UInt32 type, IOService * provider,
				void * argument);

  IOService * getAudioOut() { return fAudioOut; };
};

#endif /* _IOBSDCONSOLE_H */
