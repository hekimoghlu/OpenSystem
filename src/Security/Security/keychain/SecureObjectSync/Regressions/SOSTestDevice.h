/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include "keychain/SecureObjectSync/SOSMessage.h"
#include <utilities/SecDb.h>
#include <CoreFoundation/CFRuntime.h>

typedef struct __OpaqueSOSTestDevice *SOSTestDeviceRef;

struct __OpaqueSOSTestDevice {
    CFRuntimeBase _base;
    SecDbRef db;
    SOSDataSourceFactoryRef dsf;
    SOSDataSourceRef ds;
    CFMutableArrayRef peers;
    bool mute;
};

CFStringRef SOSMessageCopyDigestHex(SOSMessageRef message);

CFStringRef SOSTestDeviceGetID(SOSTestDeviceRef td);
void SOSTestDeviceForEachPeerID(SOSTestDeviceRef td, void(^peerBlock)(CFStringRef peerID, bool *stop));
SOSTestDeviceRef SOSTestDeviceCreateWithDb(CFAllocatorRef allocator, CFStringRef engineID, SecDbRef db);
SOSTestDeviceRef SOSTestDeviceCreateWithDbNamed(CFAllocatorRef allocator, CFStringRef engineID, CFStringRef dbName);
SOSTestDeviceRef SOSTestDeviceCreateWithTestDataSource(CFAllocatorRef allocator, CFStringRef engineID,
                                                       void(^prepop)(SOSDataSourceRef ds));
CFSetRef SOSViewsCopyTestV0Default(void);
CFSetRef SOSViewsCopyTestV2Default(void);
SOSTestDeviceRef SOSTestDeviceSetPeerIDs(SOSTestDeviceRef td, CFArrayRef peerIDs, CFIndex version, CFSetRef defaultViews);
void SOSTestDeviceDestroyEngine(CFMutableDictionaryRef testDevices);

void SOSTestDeviceForceCloseDatabase(SOSTestDeviceRef testDevice);
void SOSTestDeviceForceCloseDatabases(CFMutableDictionaryRef testDevices);

SOSTestDeviceRef SOSTestDeviceSetMute(SOSTestDeviceRef td, bool mute);
bool SOSTestDeviceIsMute(SOSTestDeviceRef td);

bool SOSTestDeviceSetEngineState(SOSTestDeviceRef td, CFDataRef derEngineState);
bool SOSTestDeviceEngineSave(SOSTestDeviceRef td, CFErrorRef *error);
bool SOSTestDeviceEngineLoad(SOSTestDeviceRef td, CFErrorRef *error);

CFDataRef SOSTestDeviceCreateMessage(SOSTestDeviceRef td, CFStringRef peerID);

bool SOSTestDeviceHandleMessage(SOSTestDeviceRef td, CFStringRef peerID, CFDataRef msgData);

void SOSTestDeviceAddGenericItem(SOSTestDeviceRef td, CFStringRef account, CFStringRef server);
void SOSTestDeviceAddGenericItemTombstone(SOSTestDeviceRef td, CFStringRef account, CFStringRef server);
void SOSTestDeviceAddGenericItemWithData(SOSTestDeviceRef td, CFStringRef account, CFStringRef server, CFDataRef data);
void SOSTestDeviceAddRemoteGenericItem(SOSTestDeviceRef td, CFStringRef account, CFStringRef server);
bool SOSTestDeviceAddGenericItems(SOSTestDeviceRef td, CFIndex count, CFStringRef account, CFStringRef server);
void SOSTestDeviceAddV0EngineStateWithData(SOSDataSourceRef ds, CFDataRef engineStateData);

CFMutableDictionaryRef SOSTestDeviceListCreate(bool realDb, CFIndex version, CFArrayRef deviceIDs,
                                               void(^prepop)(SOSDataSourceRef ds));

void SOSTestDeviceListSync(const char *name, const char *test_directive, const char *test_reason, CFMutableDictionaryRef testDevices, bool(^pre)(SOSTestDeviceRef source, SOSTestDeviceRef dest), bool(^post)(SOSTestDeviceRef source, SOSTestDeviceRef dest, SOSMessageRef message));

bool SOSTestDeviceListInSync(const char *name, const char *test_directive, const char *test_reason, CFMutableDictionaryRef testDevices);

void SOSTestDeviceListTestSync(const char *name, const char *test_directive, const char *test_reason, CFIndex version, bool use_db,
                               bool(^pre)(SOSTestDeviceRef source, SOSTestDeviceRef dest),
                               bool(^post)(SOSTestDeviceRef source, SOSTestDeviceRef dest, SOSMessageRef message), ...);
