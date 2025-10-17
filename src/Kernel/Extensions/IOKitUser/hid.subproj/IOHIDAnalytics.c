/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
//
//  IOHIDAnalyticsSupport.c
//  IOKit
//
//  Created by AB on 11/27/18.
//

#include <dlfcn.h>
#include "IOHIDAnalytics.h"
#include <CoreFoundation/CoreFoundation.h>

#define HIDAnalyticsFrameworkPath "/System/Library/PrivateFrameworks/HIDAnalytics.framework/HIDAnalytics"


#define kHIDAnalyticsEventCreate                   "HIDAnalyticsEventCreate"
#define kHIDAnalyticsEventAddHistogramField        "HIDAnalyticsEventAddHistogramField"
#define kHIDAnalyticsEventAddField                 "HIDAnalyticsEventAddField"
#define kHIDAnalyticsEventActivate                 "HIDAnalyticsEventActivate"
#define kHIDAnalyticsEventCancel                   "HIDAnalyticsEventCancel"
#define kHIDAnalyticsEventSetIntegerValueForField  "HIDAnalyticsEventSetIntegerValueForField"
#define kHIDAnalyticsEventSetStringValueForField   "HIDAnalyticsEventSetStringValueForField"
#define kHIDAnalyticsHistogramEventCreate          "HIDAnalyticsHistogramEventCreate"
#define kHIDAnalyticsHistogramEventSetIntegerValue "HIDAnalyticsHistogramEventSetIntegerValue"

typedef CFTypeRef (*HIDAnalyticsEventCreatePtr)(CFStringRef eventName, CFDictionaryRef description);
typedef void (*HIDAnalyticsEventAddHistogramFieldPtr) (CFTypeRef event, CFStringRef fieldName, IOHIDAnalyticsHistogramSegmentConfig* segments, uint8_t count);
typedef void (*HIDAnalyticsEventAddFieldPtr) (CFTypeRef event, CFStringRef fieldName);
typedef void (*HIDAnalyticsEventSetIntegerValueForFieldPtr) (CFTypeRef event, CFStringRef fieldName, uint64_t value);
typedef void (*HIDAnalyticsEventSetStringValueForFieldPtr) (CFTypeRef event, CFStringRef fieldName, CFStringRef value);
typedef void (*HIDAnalyticsEventActivatePtr) (CFTypeRef  event);
typedef void (*HIDAnalyticsEventCancelPtr) (CFTypeRef  event);
typedef CFTypeRef (*HIDAnalyticsHistogramEventCreatePtr)(CFStringRef eventName, CFDictionaryRef _Nullable description, CFStringRef fieldName,IOHIDAnalyticsHistogramSegmentConfig*  segments, CFIndex count);
typedef void (*HIDAnalyticsHistogramEventSetIntegerValuePtr)(CFTypeRef  event, uint64_t value);


static HIDAnalyticsEventCreatePtr createEventFuncPtr = NULL;
static HIDAnalyticsEventAddHistogramFieldPtr addHistogramFieldFuncPtr = NULL;
static HIDAnalyticsEventAddFieldPtr addFieldFuncPtr = NULL;
static HIDAnalyticsEventSetIntegerValueForFieldPtr setIntegerValueForFieldFuncPtr = NULL;
static HIDAnalyticsEventSetStringValueForFieldPtr setStringValueForFieldFuncPtr = NULL;
static HIDAnalyticsEventActivatePtr activateEventFuncPtr = NULL;
static HIDAnalyticsEventCancelPtr cancelEventFuncPtr = NULL;
static HIDAnalyticsHistogramEventCreatePtr createHistogramEventFuncPtr = NULL;
static HIDAnalyticsHistogramEventSetIntegerValuePtr setHistogramIntegerValueFuncPtr = NULL;

static void* HIDAnalyticsFrameworkGetSymbol( void* handle, const char* symbolName) {
    
    if (!handle) {
        return NULL;
    }
    
    return dlsym(handle, symbolName);
}

static void __loadFramework()
{
 
    static void *haHandle = NULL;
    static dispatch_once_t haOnce = 0;
    
    dispatch_once(&haOnce, ^{
        
        haHandle = dlopen(HIDAnalyticsFrameworkPath, RTLD_LAZY);
        // safe place for saving required symbols
        if (!haHandle) {
            return;
        }
        createEventFuncPtr = (HIDAnalyticsEventCreatePtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventCreate);
        
        addHistogramFieldFuncPtr = (HIDAnalyticsEventAddHistogramFieldPtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventAddHistogramField);
        
        addFieldFuncPtr = (HIDAnalyticsEventAddFieldPtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventAddField);
        
        setIntegerValueForFieldFuncPtr = (HIDAnalyticsEventSetIntegerValueForFieldPtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventSetIntegerValueForField);
        
        setStringValueForFieldFuncPtr = (HIDAnalyticsEventSetStringValueForFieldPtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventSetStringValueForField);

        activateEventFuncPtr = (HIDAnalyticsEventActivatePtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventActivate);
        
        cancelEventFuncPtr = (HIDAnalyticsEventCancelPtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsEventCancel);
        
        createHistogramEventFuncPtr = (HIDAnalyticsHistogramEventCreatePtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsHistogramEventCreate);
        
        setHistogramIntegerValueFuncPtr = (HIDAnalyticsHistogramEventSetIntegerValuePtr)HIDAnalyticsFrameworkGetSymbol(haHandle, kHIDAnalyticsHistogramEventSetIntegerValue);
        
        
    });
    
}

CFTypeRef IOHIDAnalyticsEventCreate(CFStringRef eventName, CFDictionaryRef description)
{
    
    CFTypeRef event = NULL;
    
    __loadFramework();
    
    if (!createEventFuncPtr) {
        return NULL;
    }
    
    // not needed because if framework not linked
    // dl call will fail added for handing NULL case by any means
    // in framework
    event = createEventFuncPtr(eventName, description);
    
    return event ? event : NULL;
}

CFTypeRef __nullable IOHIDAnalyticsHistogramEventCreate(CFStringRef eventName, CFDictionaryRef _Nullable description, CFStringRef fieldName, IOHIDAnalyticsHistogramSegmentConfig*  segments, CFIndex count)
{
    
    CFTypeRef event = NULL;
    
    __loadFramework();
    
    if (!createHistogramEventFuncPtr) {
        return NULL;
    }
    
    event = createHistogramEventFuncPtr(eventName, description, fieldName, segments, count);
    
    return event ? event : NULL;
}

void IOHIDAnalyticsHistogramEventSetIntegerValue(CFTypeRef  event, uint64_t value)
{
    if (!setHistogramIntegerValueFuncPtr) {
        return;
    }
    
    setHistogramIntegerValueFuncPtr(event,value);
    
}

void IOHIDAnalyticsEventAddHistogramField(CFTypeRef  event, CFStringRef  fieldName, IOHIDAnalyticsHistogramSegmentConfig*  segments, CFIndex count)
{
    if (!addHistogramFieldFuncPtr) {
        return;
    }
    
    addHistogramFieldFuncPtr(event, fieldName, segments, count);
    
}

void IOHIDAnalyticsEventSetIntegerValueForField(CFTypeRef  event, CFStringRef  fieldName, uint64_t value)
{
    if (!setIntegerValueForFieldFuncPtr) {
        return;
    }
    
    setIntegerValueForFieldFuncPtr(event, fieldName, value);
}

void IOHIDAnalyticsEventSetStringValueForField(CFTypeRef event, CFStringRef fieldName, CFStringRef value)
{
    if (!setStringValueForFieldFuncPtr) {
        return;
    }

    setStringValueForFieldFuncPtr(event, fieldName, value);
}

void IOHIDAnalyticsEventActivate(CFTypeRef event)
{
    if (!activateEventFuncPtr) {
        return;
    }
    
    activateEventFuncPtr(event);
}

void IOHIDAnalyticsEventCancel(CFTypeRef event)
{
    if (!cancelEventFuncPtr) {
        return;
    }
    
    cancelEventFuncPtr(event);
}


void IOHIDAnalyticsEventAddField(CFTypeRef  event, CFStringRef  fieldName)
{
    if (!addFieldFuncPtr) {
        return;
    }
    
    addFieldFuncPtr(event, fieldName);
}
