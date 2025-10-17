/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
// child - track a single child process and its belongings
//
#ifndef _H_AUDITEVENTS
#define _H_AUDITEVENTS

#include <security_utilities/threading.h>
#include <security_utilities/mach++.h>
#include <security_utilities/kq++.h>
#include <sys/event.h>
#include <bsm/audit_session.h>


class AuditMonitor : public Thread, public UnixPlusPlus::KQueue {
public:
	AuditMonitor(MachPlusPlus::Port relay);
	~AuditMonitor();
	
	void threadAction();

private:
	MachPlusPlus::Port mRelay;
};


#endif //_H_AUDITEVENTS
