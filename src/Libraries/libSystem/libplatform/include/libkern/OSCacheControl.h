/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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
#ifndef _OS_CACHE_CONTROL_H_
#define _OS_CACHE_CONTROL_H_

#include    <stddef.h>
#include    <sys/cdefs.h>
#include    <stdint.h>
#include    <Availability.h>

__BEGIN_DECLS


/* Functions performed by sys_cache_control(): */

/* Prepare memory for execution.  This should be called
 * after writing machine instructions to memory, before
 * executing them.  It syncs the dcache and icache.
 * On IA32 processors this function is a NOP, because
 * no synchronization is required.
 */
#define	kCacheFunctionPrepareForExecution	1

/* Flush data cache(s).  This ensures that cached data 
 * makes it all the way out to DRAM, and then removes
 * copies of the data from all processor caches.
 * It can be useful when dealing with cache incoherent
 * devices or DMA.
 */
#define	kCacheFunctionFlushDcache	2


/* perform one of the above cache functions: */
int	sys_cache_control( int function, void *start, size_t len) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
 
/* equivalent to sys_cache_control(kCacheFunctionPrepareForExecution): */
void	sys_icache_invalidate( void *start, size_t len) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);

/* equivalent to sys_cache_control(kCacheFunctionFlushDcache): */
void	sys_dcache_flush( void *start, size_t len) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);


__END_DECLS

#endif /* _OS_CACHE_CONTROL_H_ */
