/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#include "PrivateLib.h"
#include "PMConnection.h"

#ifndef _HIDEvent_h_
#define _HIDEvent_h_

__private_extern__ kern_return_t _io_pm_hid_event_report_activity(
    mach_port_t server,
    audit_token_t token,                                                        
    int         _action,
    int         *alowEvent);

__private_extern__ kern_return_t _io_pm_hid_event_copy_history(
            mach_port_t     server,
            vm_offset_t     *array_data,
            mach_msg_type_number_t  *array_dataLen,
            int             *return_val);

#endif
