/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
#ifndef _DNS_CONFIGURATION_H
#define _DNS_CONFIGURATION_H

#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>

#define DNS_CONFIGURATION_SCOPED_QUERY_KEY	CFSTR("__SCOPED_QUERY__")


__BEGIN_DECLS
typedef void (*dns_change_callback)(void);
void	dns_configuration_init		(CFBundleRef		bundle);


void	dns_configuration_monitor	(dispatch_queue_t queue,
					 dns_change_callback	callback);

Boolean	dns_configuration_set		(CFDictionaryRef	defaultResolver,
					 CFDictionaryRef	services,
					 CFArrayRef		serviceOrder,
					 CFArrayRef		multicastResolvers,
					 CFArrayRef		privateResolvers,
					 CFDictionaryRef	*globalResolver);

__END_DECLS

#endif	/* _DNS_CONFIGURATION_H */

