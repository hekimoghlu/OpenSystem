/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#include <IOKit/IOMessage.h>
#include <IOKit/IOLib.h>
#include "IOBSDConsole.h"
#include <IOKit/hidsystem/IOHIKeyboard.h>
#include <IOKit/hidsystem/IOLLEvent.h>

#define super IOService
OSDefineMetaClassAndStructors(IOBSDConsole, IOService);

// remove
bool (*playBeep)(IOService *outputStream) = 0;

//************************************************************************

bool IOBSDConsole::start(IOService * provider)
{
    OSObject *	notify;

    if (!super::start(provider))  return false;

    notify = addNotification( gIOPublishNotification,
        serviceMatching("IOHIKeyboard"),
        (IOServiceNotificationHandler) &IOBSDConsole::publishNotificationHandler,
        this, 0 );
    assert( notify );

    notify = addNotification( gIOPublishNotification,
        serviceMatching("IOAudioStream"),
        (IOServiceNotificationHandler) &IOBSDConsole::publishNotificationHandler,
        this, this );
    assert( notify );

    return( true );
}

bool IOBSDConsole::publishNotificationHandler(
			    IOBSDConsole * self,
                            void * ref,
                            IOService * newService )

{
    IOHIKeyboard *	keyboard = 0;
    IOService *		audio = 0;

    if( ref) {
        audio = OSDynamicCast(IOService, newService->metaCast("IOAudioStream"));
        if (audio != 0) {
            OSNumber *out = newService->copyProperty("Out");
            if (OSDynamicCast(OSNumber, out)) {
                if (out->unsigned8BitValue() == 1) {
                    self->fAudioOut = newService;
                }
            }
            OSSafeReleaseNULL(out);
        }
    } else {
	audio = 0;
        keyboard = OSDynamicCast( IOHIKeyboard, newService );

        if( keyboard && self->attach( keyboard )) {
            self->arbitrateForKeyboard( keyboard );
        }
    }

    return true;
}

//************************************************************************
// Keyboard client stuff
//************************************************************************

void IOBSDConsole::arbitrateForKeyboard( IOHIKeyboard * nub )
{
  nub->open(this, 0, 0,
	(KeyboardEventCallback)keyboardEvent, 
        (KeyboardSpecialEventCallback) 0, 
        (UpdateEventFlagsCallback)updateEventFlags);
  // failure can be expected if the HID system already has it
}

IOReturn IOBSDConsole::message(UInt32 type, IOService * provider,
				void * argument)
{
  IOReturn     status = kIOReturnSuccess;

  switch (type)
  {
    case kIOMessageServiceIsTerminated:
    case kIOMessageServiceIsRequestingClose:
      provider->close( this );
      break;

    case kIOMessageServiceWasClosed:
      arbitrateForKeyboard( (IOHIKeyboard *) provider );
      break;

    default:
      status = super::message(type, provider, argument);
      break;
  }

  return status;
}

extern "C" {
  void cons_cinput( char c);
}
//#warning REMOVE cons_cinput DECLARATION FROM HERE

void IOBSDConsole::keyboardEvent(OSObject * target,
          /* eventType */        unsigned   eventType,
          /* flags */            unsigned   flags,
          /* keyCode */          unsigned   /* key */,
          /* charCode */         unsigned   charCode,
          /* charSet */          unsigned   charSet,
          /* originalCharCode */ unsigned   /* origCharCode */,
          /* originalCharSet */  unsigned   /* origCharSet */,
          /* keyboardType */ 	 unsigned   /* keyboardType */,
          /* repeat */           bool       /* repeat */,
          /* atTime */           AbsoluteTime /* ts */,
                                 OSObject * sender,
                                 void *     refcon)
{
    static const char cursorCodes[] = { 'D', 'A', 'C', 'B' };

    // declare that there is user activity
    getPMRootDomain()->requestUserActive(this, "IOBSDConsole::keyboardEvent");

    if( (eventType == NX_KEYDOWN) && ((flags & NX_ALTERNATEMASK) != NX_ALTERNATEMASK)) {
        if( (charSet == NX_SYMBOLSET)
            && (charCode >= 0xac) && (charCode <= 0xaf)) {
            cons_cinput( '\033');
            cons_cinput( 'O');
            charCode = cursorCodes[ charCode - 0xac ];
        }
        cons_cinput( charCode);
    }
}

void IOBSDConsole::updateEventFlags(OSObject * /*target*/, 
                                    unsigned /*flags*/,
                                    OSObject * /*sender*/,
                                    void *     /*refcon*/)
{
  return;
}


