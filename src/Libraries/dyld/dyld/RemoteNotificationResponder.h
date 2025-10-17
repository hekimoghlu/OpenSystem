/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#ifndef RemoteNotificationResponder_h
#define RemoteNotificationResponder_h

#if !TARGET_OS_SIMULATOR
#include <mach/mach.h>

namespace dyld4 {
struct RemoteNotificationResponder {
    RemoteNotificationResponder(const RemoteNotificationResponder&) = delete;
    RemoteNotificationResponder(RemoteNotificationResponder&&) = delete;
    RemoteNotificationResponder(mach_port_t notifyPortValue);
    ~RemoteNotificationResponder();
    void notifyMonitorOfImageListChanges(bool unloading, unsigned imageCount, const struct mach_header* loadAddresses[], const char* imagePaths[], uint64_t lastUpdateTime);
    void notifyMonitorOfMainCalled();
    void notifyMonitorOfDyldBeforeInitializers();
    void sendMessage(mach_msg_id_t msgId, mach_msg_size_t sendSize, mach_msg_header_t* buffer);
    bool const active() const;
    void blockOnSynchronousEvent(uint32_t event);
    
    enum { DYLD_PROCESS_INFO_NOTIFY_MAGIC = 0x49414E46 };

private:
    mach_port_t             _namesArray[8] = {0};
    mach_port_name_array_t  _names = (mach_port_name_array_t)&_namesArray[0];
    mach_msg_type_number_t  _namesCnt = 8;
    vm_size_t               _namesSize = 0;
};

}; /* namespace dyld4 */

#endif /* !TARGET_OS_SIMULATOR */
#endif /* RemoteNotificationResponder_h */
