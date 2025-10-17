/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#pragma once

#if defined __has_include && __has_include(<CoreFoundation/CFPriv.h>)
#include <CoreMedia/CMBaseObject.h>
#else

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kCMBaseObject_ClassVersion_1 = 1,
    kCMBaseObject_ClassVersion_2 = 2,
    kCMBaseObject_ClassVersion_3 = 3
};

enum {
    kCMBaseObject_ProtocolVersion_1 = 1
};

enum {
    kCMBaseObjectError_ValueNotAvailable = -12783,
};

typedef struct OpaqueCMBaseObject *CMBaseObjectRef;
typedef struct OpaqueCMBaseClass *CMBaseClassID;
typedef struct OpaqueCMBaseProtocol *CMBaseProtocolID;

typedef OSStatus (*CMBaseObjectCopyPropertyFunction)(CMBaseObjectRef, CFStringRef propertyKey, CFAllocatorRef, void *propertyValueOut);
typedef OSStatus (*CMBaseObjectSetPropertyFunction)(CMBaseObjectRef object, CFStringRef propertyKey, CFTypeRef  propertyValue);

#pragma pack(push)
#pragma pack()
typedef struct {
    CMBaseClassVersion version;
    CFStringRef (*copyProtocolDebugDescription)(CMBaseObjectRef);
} CMBaseProtocol;
#pragma pack(pop)

#pragma pack(push)
#pragma pack()
struct CMBaseProtocolVTable {
    const struct OpaqueCMBaseProtocolVTableReserved *reserved;
    const CMBaseProtocol *baseProtocol;
};
#pragma pack(pop)

typedef struct CMBaseProtocolTableEntry {
    CMBaseProtocolID (*getProtocolID)(void);
    CMBaseProtocolVTable *protocolVTable;
} CMBaseProtocolTableEntry;

struct CMBaseProtocolTable {
    uint32_t version;
    uint32_t numSupportedProtocols;
    CMBaseProtocolTableEntry * supportedProtocols;
};

#pragma pack(push, 4)
typedef struct {
   CMBaseClassVersion version;
   size_t derivedStorageSize;

   Boolean (*equal)(CMBaseObjectRef, CMBaseObjectRef);
   OSStatus (*invalidate)(CMBaseObjectRef);
   void (*finalize)(CMBaseObjectRef);
   CFStringRef (*copyDebugDescription)(CMBaseObjectRef);

   CMBaseObjectCopyPropertyFunction copyProperty;
   CMBaseObjectSetPropertyFunction setProperty;
   OSStatus (*notificationBarrier)(CMBaseObjectRef);
   const CMBaseProtocolTable *protocolTable;
} CMBaseClass;
#pragma pack(pop)

#pragma pack(push)
#pragma pack()
typedef struct {
    const struct OpaqueCMBaseVTableReserved *reserved;
    const CMBaseClass *baseClass;
} CMBaseVTable;
#pragma pack(pop)

CM_EXPORT OSStatus CMDerivedObjectCreate(CFAllocatorRef, const CMBaseVTable*, CMBaseClassID, CMBaseObjectRef*);
CM_EXPORT void* CMBaseObjectGetDerivedStorage(CMBaseObjectRef);
CM_EXPORT const CMBaseVTable* CMBaseObjectGetVTable(CMBaseObjectRef);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __has_include && __has_include(<CoreFoundation/CFPriv.h>)
