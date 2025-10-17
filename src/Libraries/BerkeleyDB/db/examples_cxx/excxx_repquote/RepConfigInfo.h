/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#include <db_cxx.h>

// Chainable struct used to store host information.
typedef struct RepHostInfoObj{
	char* host;
	int port;
	bool peer; // only relevant for "other" hosts
	RepHostInfoObj* next; // used for chaining multiple "other" hosts.
} REP_HOST_INFO;

class RepConfigInfo {
public:
	RepConfigInfo();
	virtual ~RepConfigInfo();

	void addOtherHost(char* host, int port, bool peer);
public:
	u_int32_t start_policy;
	char* home;
	bool got_listen_address;
	REP_HOST_INFO this_host;
	int totalsites;
	int priority;
	bool verbose;
	// used to store a set of optional other hosts.
	REP_HOST_INFO *other_hosts;
};
