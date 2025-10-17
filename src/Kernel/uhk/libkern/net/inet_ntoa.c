/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include <sys/param.h>
#include <sys/systm.h>

#include <netinet/in.h>

char *
inet_ntoa(struct in_addr ina)
{
	static char buf[4 * sizeof "123"];
	unsigned char *ucp = (unsigned char *)&ina;

	snprintf(buf, sizeof(buf), "%d.%d.%d.%d",
	    ucp[0] & 0xff,
	    ucp[1] & 0xff,
	    ucp[2] & 0xff,
	    ucp[3] & 0xff);
	return buf;
}

char *
inet_ntoa_r(struct in_addr ina, char *buf, size_t buflen)
{
	unsigned char *ucp = (unsigned char *)&ina;

	snprintf(buf, buflen, "%d.%d.%d.%d",
	    ucp[0] & 0xff,
	    ucp[1] & 0xff,
	    ucp[2] & 0xff,
	    ucp[3] & 0xff);
	return buf;
}
