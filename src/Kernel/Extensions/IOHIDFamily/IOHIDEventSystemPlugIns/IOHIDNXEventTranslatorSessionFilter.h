/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#if COREFOUNDATION_CFPLUGINCOM_SEPARATE
#include <CoreFoundation/CFPlugInCOM.h>
#endif
#include <IOKit/hidsystem/event_status_driver.h>
#include <mach/mach_time.h>
#include <queue>
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <IOKit/pwr_mgt/IOPMLibDefs.h>
#include <IOKit/hid/IOHIDLibPrivate.h>
#include  <shared_mutex>
#include "CF.h"
#import <Foundation/Foundation.h>
#import <SkyLight/SLSDisplayManager.h>


#define kIOHIDPowerOnThresholdMS                    (500)            // 1/2 second
#define kIOHIDDisplaySleepAbortThresholdMS          (5000)           // 5 seconds
#define kIOHIDDisplaySleepPolicyUpgradeThresholdMS  (25)
#define LOG_MAX_ENTRIES                             (50)
#define kIOHIDDisplayWakeAbortThresholdMS           (50)
#define kIOHIDDeclareActivityThresholdMS            250

struct LogEntry {
    struct timeval          time;
    IOHIDEventSenderID      serviceID;
    IOHIDEventPolicyValue   policy;
    IOHIDEventType          eventType;
    uint64_t                timestamp;
};

struct LogNXEventEntry {
    struct timeval         time;
    IOHIDEventPolicyValue  policy;
    int32_t                senderPID;
    uint32_t               nxEventType;
    uint64_t               timestamp;
};

class IOHIDNXEventTranslatorSessionFilter
{
public:
    IOHIDNXEventTranslatorSessionFilter(CFUUIDRef factoryID);
    ~IOHIDNXEventTranslatorSessionFilter();
    HRESULT QueryInterface( REFIID iid, LPVOID *ppv );
    ULONG AddRef();
    ULONG Release();
    
    IOHIDEventRef filter(IOHIDServiceRef sender, IOHIDEventRef event);
    boolean_t open(IOHIDSessionRef session, IOOptionBits options);
    void close(IOHIDSessionRef session, IOOptionBits options);
    void registerService(IOHIDServiceRef service);
    void unregisterService(IOHIDServiceRef service);
    void scheduleWithDispatchQueue(dispatch_queue_t queue);
    void unscheduleFromDispatchQueue(dispatch_queue_t queue);
    void setPropertyForClient (CFStringRef key, CFTypeRef property, CFTypeRef client);
    CFTypeRef getPropertyForClient (CFStringRef key, CFTypeRef client);
  
private:

    IOHIDSessionFilterPlugInInterface *_sessionInterface;
    CFUUIDRef                       _factoryID;
    UInt32                          _refCount;
    dispatch_queue_t                _dispatch_queue;

    uint32_t                        _globalModifiers;
    NXEventHandle                   _hidSystem;
    IOHIDPointerEventTranslatorRef  _translator;
    io_connect_t                    _powerConnect;
    io_object_t                     _powerNotifier;
    IONotificationPortRef           _powerPort;
    uint32_t                        _powerState;
    uint32_t                        _powerOnThresholdEventCount;
    uint32_t                        _powerOnThreshold;
    uint32_t                        _displayState;
    uint32_t                        _displaySleepAbortThreshold;
    uint32_t                        _displayWakeAbortThreshold;
    static IOPMAssertionID          _AssertionID;
    CFMutableDictionaryRefWrap      _assertionNames;
    uint64_t                        _previousEventTime;
    uint64_t                        _declareActivityThreshold;
    dispatch_queue_t                _updateActivityQueue;
    dispatch_queue_t                _modifiersQueue;
  
    CFMutableDictionaryRefWrap      _modifiers;
    CFMutableDictionaryRefWrap      _companions;
    CFMutableSetRefWrap             _keyboards;
    CFMutableSetRefWrap             _reportModifiers;
    CFMutableSetRefWrap             _updateModifiers;
    IOHIDServiceRef                 _dfr;
    bool                            _isTranslationEnabled;
    
    uint64_t    _powerStateChangeTime;
    uint64_t    _displayStateChangeTime;
    
    IOHIDSimpleQueueRef             _displayLog;
    IOHIDSimpleQueueRef             _nxEventLog;
    
private:

    static IOHIDSessionFilterPlugInInterface sIOHIDNXEventTranslatorSessionFilterFtbl;
    static HRESULT QueryInterface( void *self, REFIID iid, LPVOID *ppv );
    static ULONG AddRef( void *self );
    static ULONG Release( void *self );
    
    static IOHIDEventRef filter(void * self, IOHIDServiceRef sender, IOHIDEventRef event);

    static boolean_t open(void * self, IOHIDSessionRef inSession, IOOptionBits options);
    static void close(void * self, IOHIDSessionRef inSession, IOOptionBits options);
    static void registerService(void * self, IOHIDServiceRef service);
    static void unregisterService(void * self, IOHIDServiceRef service);
  
    static void scheduleWithDispatchQueue(void * self, dispatch_queue_t queue);
    static void unscheduleFromDispatchQueue(void * self, dispatch_queue_t queue);
    static void setPropertyForClient (void * self, CFStringRef key, CFTypeRef property, CFTypeRef client);
    static CFTypeRef getPropertyForClient (void * self, CFStringRef key, CFTypeRef client);

    void updateModifiers();
    void updateButtons();
    void updateActivity (bool active);
    void updateDisplayLog(IOHIDEventSenderID serviceID, IOHIDEventPolicyValue policy, IOHIDEventType eventType, uint64_t timestamp);
    void updateNXEventLog(IOHIDEventPolicyValue policy, IOHIDEventRef event, uint64_t timestamp);
    
    IOHIDServiceRef getCompanionService(IOHIDServiceRef service);
    
    IOHIDEventRef powerStateFilter (IOHIDEventRef  event);
    static void powerNotificationCallback (void * refcon, io_service_t	service, uint32_t messageType, void * messageArgument);
    void powerNotificationCallback (io_service_t	service, uint32_t messageType, void * messageArgument);
  
    void displayNotificationCallback (SLSDisplayPowerStateNotificationType state);
    IOHIDEventRef displayStateFilter (IOHIDServiceRef sender, IOHIDEventRef  event);
  
    boolean_t shouldCancelEvent (IOHIDEventRef  event);
    boolean_t resetStickyKeys(IOHIDEventRef event);
 
    void serialize (CFMutableDictionaryRef dict) const;

    CFStringRef createNXEventActivityString(IOHIDServiceRef sender, NXEventExt * nxEvent);
    CFStringRef createHIDEventActivityString(IOHIDServiceRef sender, IOHIDEventRef event);
  
private:
    IOHIDNXEventTranslatorSessionFilter();
    IOHIDNXEventTranslatorSessionFilter(const IOHIDNXEventTranslatorSessionFilter &);
    IOHIDNXEventTranslatorSessionFilter &operator=(const IOHIDNXEventTranslatorSessionFilter &);
};
