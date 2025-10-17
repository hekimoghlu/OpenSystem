/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
 * EventSrcPCKeyboard.h - PC Keyboard EventSrc subclass definition
 *
 * HISTORY
 * 28 Aug 1992    Joe Pasqua
 *      Created. 
 */

#ifndef _IOHIKEYBOARD_H
#define _IOHIKEYBOARD_H

#include <IOKit/hidsystem/IOHIDevice.h>
#include <IOKit/hidsystem/IOHIKeyboardMapper.h>

/* Start Action Definitions */

/*
 * HISTORICAL NOTE:
 *   The following entry points were part of the IOHIKeyboardEvents
 *   protocol.
 */

typedef void (*KeyboardEventAction)(       OSObject * target,
                    /* eventFlags  */      unsigned   eventType,
                    /* flags */            unsigned   flags,
                    /* keyCode */          unsigned   key,
                    /* charCode */         unsigned   charCode,
                    /* charSet */          unsigned   charSet,
                    /* originalCharCode */ unsigned   origCharCode,
                    /* originalCharSet */  unsigned   origCharSet,
                    /* keyboardType */     unsigned   keyboardType,
                    /* repeat */           bool       repeat,
                    /* atTime */           AbsoluteTime ts);

typedef void (*KeyboardSpecialEventAction)(OSObject * target,
                    /* eventType */        unsigned   eventType,
                    /* flags */            unsigned   flags,
                    /* keyCode */          unsigned   key,
                    /* specialty */        unsigned   flavor,
                    /* source id */        UInt64     guid,
                    /* repeat */           bool       repeat,
                    /* atTime */           AbsoluteTime ts);

typedef void (*UpdateEventFlagsAction)(    OSObject * target,
                     /* flags */           unsigned   flags);

/* Event Callback Definitions */

typedef void (*KeyboardEventCallback)(
                    /* target */           OSObject * target,
                    /* eventFlags  */      unsigned   eventType,
                    /* flags */            unsigned   flags,
                    /* keyCode */          unsigned   key,
                    /* charCode */         unsigned   charCode,
                    /* charSet */          unsigned   charSet,
                    /* originalCharCode */ unsigned   origCharCode,
                    /* originalCharSet */  unsigned   origCharSet,
                    /* keyboardType */     unsigned   keyboardType,
                    /* repeat */           bool       repeat,
                    /* atTime */           AbsoluteTime ts,
                    /* sender */           OSObject * sender,
                    /* refcon */           void *     refcon);

typedef void (*KeyboardSpecialEventCallback)(
                    /* target */           OSObject * target,
                    /* eventType */        unsigned   eventType,
                    /* flags */            unsigned   flags,
                    /* keyCode */          unsigned   key,
                    /* specialty */        unsigned   flavor,
                    /* source id */        UInt64     guid,
                    /* repeat */           bool       repeat,
                    /* atTime */           AbsoluteTime ts,
                    /* sender */           OSObject * sender,
                    /* refcon */           void *     refcon);
                    
typedef void (*UpdateEventFlagsCallback)(
                    /* target */           OSObject * target,
                    /* flags */            unsigned   flags,
                    /* sender */           OSObject * sender,
                    /* refcon */           void *     refcon);

/* End Action Definitions */

/* Default key repeat parameters */
#define EV_DEFAULTINITIALREPEAT 500000000ULL    // 1/2 sec in nanoseconds
#define EV_DEFAULTKEYREPEAT     83333333ULL     // 1/12 sec in nanoseconds
#define EV_MINKEYREPEAT         16700000ULL     // 1/60 sec

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIKeyboard : public IOHIDevice
#else
class IOHIKeyboard : public IOHIDevice
#endif
{
    OSDeclareDefaultStructors(IOHIKeyboard);

    friend class IOHIDKeyboardDevice;
    friend class IOHIDKeyboardEventDevice;
    friend class IOHIDKeyboard;
    friend class IOHIDConsumer;

protected:
    IOLock *	         _deviceLock;	// Lock for all device access
    IOHIKeyboardMapper * _keyMap;	// KeyMap instance

    // The following fields describe the kind of keyboard
    UInt32		_interfaceType;
    UInt32		_deviceType;

    // The following fields describe the state of the keyboard
    UInt32 *	_keyState;		// kbdBitVector
    IOByteCount _keyStateSize;		// kbdBitVector allocated size
    unsigned	_eventFlags;		// Current eventFlags
    bool	_alphaLock;		// true means alpha lock is on
    bool	_numLock;		// true means num lock is on
    bool	_charKeyActive;		// true means char gen. key active

    // The following fields are used in performing key repeats
    bool 	_isRepeat;		// true means we're generating repeat
    unsigned	_codeToRepeat;		// What we are repeating
    bool	_calloutPending;	// true means we've sched. a callout
    AbsoluteTime	_lastEventTime;		// Time last event was dispatched
    AbsoluteTime	_downRepeatTime;	// Time when we should next repeat
    AbsoluteTime	_keyRepeat;		// Delay between key repeats
    AbsoluteTime	_initialKeyRepeat;	// Delay before initial key repeat
    UInt64		_guid;

    OSObject *                 _keyboardEventTarget;
    KeyboardEventAction        _keyboardEventAction;
    OSObject *                 _keyboardSpecialEventTarget;
    KeyboardSpecialEventAction _keyboardSpecialEventAction;
    OSObject *                 _updateEventFlagsTarget;
    UpdateEventFlagsAction     _updateEventFlagsAction;
    
    UInt16      _lastUsagePage;
    UInt16      _lastUsage;

protected:
  virtual void dispatchKeyboardEvent(unsigned int keyCode,
		     /* direction */ bool         goingDown,
		     /* timeStamp */ AbsoluteTime time);
    void    setLastPageAndUsage(UInt16 usagePage, UInt16 usage);
    void    getLastPageAndUsage(UInt16 &usagePage, UInt16 &usage);
    void    clearLastPageAndUsage();

public:
  virtual bool init(OSDictionary * properties = 0) APPLE_KEXT_OVERRIDE;
  virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
  virtual void stop(IOService * provider) APPLE_KEXT_OVERRIDE;
  virtual void free(void) APPLE_KEXT_OVERRIDE;

  virtual bool open(IOService *                client,
		    IOOptionBits	       options,
                    KeyboardEventAction        keAction,
                    KeyboardSpecialEventAction kseAction,
                    UpdateEventFlagsAction     uefAction);
                    
  bool open(        IOService *                  client,
		    IOOptionBits	         options,
                    void *,
                    KeyboardEventCallback        keCallback,
                    KeyboardSpecialEventCallback kseCallback,
                    UpdateEventFlagsCallback     uefCallback);

  virtual void close(IOService * client, IOOptionBits ) APPLE_KEXT_OVERRIDE;

  virtual IOReturn message( UInt32 type, IOService * provider,
                    void * argument = 0 ) APPLE_KEXT_OVERRIDE;

  virtual IOHIDKind hidKind( void ) APPLE_KEXT_OVERRIDE;
  virtual bool 	    updateProperties( void ) APPLE_KEXT_OVERRIDE;
  virtual IOReturn  setParamProperties(OSDictionary * dict) APPLE_KEXT_OVERRIDE;
  virtual IOReturn  setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;
  
  inline  bool	    isRepeat() {return _isRepeat;}

protected: // for subclasses to implement
  virtual const unsigned char * defaultKeymapOfLength(UInt32 * length);
  virtual void setAlphaLockFeedback(bool val);
  virtual void setNumLockFeedback(bool val);
  virtual UInt32 maxKeyCodes();


private:
  virtual bool resetKeyboard();
  virtual void scheduleAutoRepeat();
  static void _autoRepeat(void * arg, void *);
  virtual void autoRepeat();
  virtual void setRepeat(unsigned eventType, unsigned keyCode);
  void setRepeatMode(bool repeat);
  static void _createKeyboardNub(thread_call_param_t param0, thread_call_param_t param1);

/*
 * HISTORICAL NOTE:
 *   The following methods were part of the KeyMapDelegate protocol;
 *   the declarations have now been merged directly into this class.
 */

public:
  virtual void keyboardEvent(unsigned eventType,
      /* flags */            unsigned flags,
      /* keyCode */          unsigned keyCode,
      /* charCode */         unsigned charCode,
      /* charSet */          unsigned charSet,
      /* originalCharCode */ unsigned origCharCode,
      /* originalCharSet */  unsigned origCharSet);

  virtual void keyboardSpecialEvent(unsigned eventType,
		    /* flags */     unsigned flags,
		    /* keyCode */   unsigned keyCode,
		    /* specialty */ unsigned flavor);

  virtual void updateEventFlags(unsigned flags); // Does not generate events

  virtual unsigned eventFlags();           // Global event flags
  virtual unsigned deviceFlags();          // per-device event flags
  virtual void setDeviceFlags(unsigned flags); // Set device event flags
  virtual bool alphaLock();                // current alpha-lock state
  virtual void setAlphaLock(bool val);     // Set current alpha-lock state
  virtual bool numLock();                
  virtual void setNumLock(bool val);     
  virtual bool charKeyActive();            // Is a character gen. key down?
  virtual void setCharKeyActive(bool val); // Note that a char gen key is down.
  virtual bool doesKeyLock(unsigned key);  //does key lock physically
  virtual unsigned getLEDStatus();  //check hardware for LED status

private:
  static void _keyboardEvent( IOHIKeyboard * self,
			     unsigned   eventType,
      /* flags */            unsigned   flags,
      /* keyCode */          unsigned   key,
      /* charCode */         unsigned   charCode,
      /* charSet */          unsigned   charSet,
      /* originalCharCode */ unsigned   origCharCode,
      /* originalCharSet */  unsigned   origCharSet,
      /* keyboardType */     unsigned   keyboardType,
      /* repeat */           bool       repeat,
      /* atTime */           AbsoluteTime ts);
  static void _keyboardSpecialEvent( 	
                             IOHIKeyboard * self,
                             unsigned   eventType,
        /* flags */          unsigned   flags,
        /* keyCode  */       unsigned   key,
        /* specialty */      unsigned   flavor,
        /* guid */           UInt64     guid,
        /* repeat */         bool       repeat,
        /* atTime */         AbsoluteTime ts);
        
  static void _updateEventFlags( IOHIKeyboard * self,
				unsigned flags);  /* Does not generate events */

};

#endif /* !_IOHIKEYBOARD_H */
