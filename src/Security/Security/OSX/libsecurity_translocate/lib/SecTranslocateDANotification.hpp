/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
/* Purpose: This header defines the Disk Arbitration monitor for translocation */

#ifndef SecTranslocateDANotification_hpp
#define SecTranslocateDANotification_hpp

#include <dispatch/dispatch.h>
#include <DiskArbitration/DiskArbitration.h>

namespace Security {
namespace SecTranslocate {
    
class DANotificationMonitor {
public:
    DANotificationMonitor(dispatch_queue_t q); //throws
    ~DANotificationMonitor();
private:
    DANotificationMonitor() = delete;
    DANotificationMonitor(const DANotificationMonitor& that) = delete;
    
    DASessionRef diskArbitrationSession;
};
    
} //namespace SecTranslocate
} //namespace Security


#endif /* SecTranslocateDANotification_hpp */
