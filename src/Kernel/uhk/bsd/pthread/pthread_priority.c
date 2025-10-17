/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include <pthread/priority_private.h>

#ifndef QOS_MIN_RELATIVE_PRIORITY // from <sys/qos.h> in userspace
#define QOS_MIN_RELATIVE_PRIORITY -15
#endif

pthread_priority_compact_t
_pthread_priority_normalize(pthread_priority_t pp)
{
	if (pp & _PTHREAD_PRIORITY_EVENT_MANAGER_FLAG) {
		return _PTHREAD_PRIORITY_EVENT_MANAGER_FLAG;
	}
	if (_pthread_priority_has_qos(pp)) {
		int relpri = _pthread_priority_relpri(pp);
		if (relpri > 0 || relpri < QOS_MIN_RELATIVE_PRIORITY) {
			pp |= _PTHREAD_PRIORITY_PRIORITY_MASK;
		}
		return pp & (_PTHREAD_PRIORITY_OVERCOMMIT_FLAG |
		       _PTHREAD_PRIORITY_FALLBACK_FLAG |
		       _PTHREAD_PRIORITY_QOS_CLASS_MASK |
		       _PTHREAD_PRIORITY_PRIORITY_MASK);
	}
	return _pthread_unspecified_priority();
}

pthread_priority_compact_t
_pthread_priority_normalize_for_ipc(pthread_priority_t pp)
{
	if (_pthread_priority_has_qos(pp)) {
		int relpri = _pthread_priority_relpri(pp);
		if (relpri > 0 || relpri < QOS_MIN_RELATIVE_PRIORITY) {
			pp |= _PTHREAD_PRIORITY_PRIORITY_MASK;
		}
		return pp & (_PTHREAD_PRIORITY_QOS_CLASS_MASK |
		       _PTHREAD_PRIORITY_PRIORITY_MASK);
	}
	return _pthread_unspecified_priority();
}

pthread_priority_compact_t
_pthread_priority_combine(pthread_priority_t base_pp, thread_qos_t qos)
{
	if (base_pp & _PTHREAD_PRIORITY_EVENT_MANAGER_FLAG) {
		return _PTHREAD_PRIORITY_EVENT_MANAGER_FLAG;
	}

	if (base_pp & _PTHREAD_PRIORITY_FALLBACK_FLAG) {
		if (!qos) {
			return (pthread_priority_compact_t)base_pp;
		}
	} else if (qos < _pthread_priority_thread_qos(base_pp)) {
		return (pthread_priority_compact_t)base_pp;
	}

	return _pthread_priority_make_from_thread_qos(qos, 0,
	           base_pp & _PTHREAD_PRIORITY_OVERCOMMIT_FLAG);
}
