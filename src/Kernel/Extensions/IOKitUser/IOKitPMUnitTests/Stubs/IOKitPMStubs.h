/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
//  IOKitPMStubs.h
//  IOKitUser
//
//  Created by Faramola Isiaka on 7/15/21.
//

#ifndef IOKitPMStubs_h
#define IOKitPMStubs_h

#include <IOKit/IOKitLib.h>
#include "EnergyTraceStubs.h"
#include "DispatchStubs.h"
#include "NotifyStubs.h"
#include "XPCStubs.h"

// privateLib
dispatch_queue_t  getPMQueue(void);
io_registry_entry_t  getPMRootDomainRef(void);
IOReturn _pm_connect(mach_port_t *newConnection);
IOReturn _pm_disconnect(mach_port_t connection);

// all from powermanagement.h
kern_return_t io_pm_assertion_activity_aggregate(mach_port_t server, vm_offset_t *statsData, mach_msg_type_number_t *statsDataCnt, int *return_code);
kern_return_t io_pm_assertion_activity_log(mach_port_t server, vm_offset_t *log, mach_msg_type_number_t *logCnt, uint32_t *entryCnt, uint32_t *overflow, int *return_code);
kern_return_t io_pm_assertion_copy_details(mach_port_t server, int assertion_id, int whichData, vm_offset_t props, mach_msg_type_number_t propsCnt, vm_offset_t *assertions, mach_msg_type_number_t *assertionsCnt, int *return_val);
kern_return_t io_pm_assertion_create(mach_port_t server, vm_offset_t props, mach_msg_type_number_t propsCnt, int *assertion_id, int *disableAppSleep, int *enTrIntensity, int *return_code);
kern_return_t io_pm_assertion_retain_release(mach_port_t server, int assertion_id, int action, int *retainCnt, int *disableAppSleep, int *enableAppSleep, int *return_code);
kern_return_t io_pm_assertion_set_properties(mach_port_t server, int assertion_id, vm_offset_t props, mach_msg_type_number_t propsCnt, int *disableAppSleep, int *enableAppSleep, int *enTrIntensity, int *return_code);
kern_return_t io_pm_declare_network_client_active(mach_port_t server, vm_offset_t props, mach_msg_type_number_t propsCnt, int *assertion_id, int *disableAppSleep, int *return_code);
kern_return_t io_pm_declare_system_active(mach_port_t server, int *state, vm_offset_t props, mach_msg_type_number_t propsCnt, int *assertion_id, int *return_code);
kern_return_t io_pm_declare_user_active(mach_port_t server, int user_type, vm_offset_t props, mach_msg_type_number_t propsCnt, int *assertion_id, int *disableAppSleep, int *return_code);
kern_return_t io_pm_set_exception_limits(mach_port_t server, vm_offset_t props, mach_msg_type_number_t propsCnt, int *return_code);
kern_return_t io_pm_set_value_int(mach_port_t server, int selector, int value, int *return_val);

void IOKitPMStubsTeardown(void);
#endif /* IOKitPMStubs_h */
