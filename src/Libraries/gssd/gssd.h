/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#ifndef __GSSD_H__
#define __GSSD_H__ 1
#include <os/log.h>

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

#ifndef maximum
#define	maximum(a, b) (((a)>(b))?(a):(b))
#endif

#ifndef minimum
#define	minimum(a, b) (((a)<(b))?(a):(b))
#endif

#define CAST(T,x) (T)(uintptr_t)(x)

#define str_to_buf(s, b) do { \
	(b)->value = (s); (b)->length = strlen(s) + 1; \
	} while (0)

#define Fatal(fmt, ...) fatal("%s: %d: " fmt, __func__, __LINE__,## __VA_ARGS__)
#define Debug(fmt, ...) gssd_log(OS_LOG_TYPE_DEBUG, "%s: %d: " fmt, __func__, __LINE__,## __VA_ARGS__)
#define DEBUG(level, ...) do {\
	if (get_debug_level() >= level) {\
		Debug(__VA_ARGS__); \
	}\
} while (0)

#define HEXDUMP(level, ...) do {\
	if (get_debug_level() >= level-2) {\
		HexDump(__VA_ARGS__); \
	}\
} while (0)

#define Info(fmt, ...) do {\
	if (get_debug_level() > 1) {\
		gssd_log(OS_LOG_TYPE_INFO, "%s: %d: " fmt, __func__, __LINE__,## __VA_ARGS__); \
	} else { \
		gssd_log(OS_LOG_TYPE_INFO, "%s: " fmt, __func__,## __VA_ARGS__); \
	}\
} while (0)

#define Notice(fmt, ...) do {\
	if (get_debug_level() > 1) {\
		gssd_log(OS_LOG_TYPE_DEFAULT, "%s: %d: " fmt, __func__, __LINE__,## __VA_ARGS__); \
	} else { \
		gssd_log(OS_LOG_TYPE_DEFAULT, fmt,## __VA_ARGS__); \
	}\
} while (0)

#define Log(fmt, ...) do {\
	if (get_debug_level()) {\
		gssd_log(OS_LOG_TYPE_ERROR, "%s: %d: " fmt, __func__, __LINE__,## __VA_ARGS__); \
	} else { \
		gssd_log(OS_LOG_TYPE_ERROR, fmt,## __VA_ARGS__); \
	}\
} while (0)


__BEGIN_DECLS

extern char *buf_to_str(gss_buffer_t);
extern void gssd_enter(void *);
extern void gssd_remove(void *);
extern int gssd_check(void *);
extern char *oid_name(gss_OID);
extern char *gss_strerror(gss_OID, uint32_t, uint32_t);
extern void __attribute__((noreturn)) fatal(const char *, ...);
extern void gssd_log(os_log_type_t, const char *, ...);
extern void HexDump(const char *, size_t);
extern int traced(void);
int in_foreground(int);
extern void set_debug_level(int);
extern int get_debug_level(void);
extern int make_lucid_stream(gss_krb5_lucid_context_v1_t *, size_t *, void **);
__END_DECLS

#endif
