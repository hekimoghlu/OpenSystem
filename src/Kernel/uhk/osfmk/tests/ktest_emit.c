/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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
#include <tests/ktest_internal.h>
#include <kern/misc_protos.h>
#include <kern/debug.h>

#define EMIT(buf, size) do {\
	console_write(buf, size); \
	} while(0)

/* TODO: intelligently truncate messages if possible */
#define BOUNDS_CHECK_AND_UPDATE(ret, size) do {\
	if(ret < 0 || ret >= size) {\
	        panic("Internal ktest error in %s", __func__);\
	}\
	size -= ret;\
	msg += ret;\
} while(0)

int vsnprintf(char *, size_t, const char *, va_list);

void
ktest_emit_start(void)
{
	char str[] = "\n[KTEST]\tSTART\t" KTEST_VERSION_STR "\n";
	EMIT((char *)&str[0], sizeof(str) - 1);
}

void
ktest_emit_finish(void)
{
	char str[] = "\n[KTEST]\tFINISH\n";
	EMIT((char *)&str[0], sizeof(str) - 1);
}

void
ktest_emit_testbegin(const char * test_name)
{
	char * msg = ktest_output_buf;
	int size = sizeof(ktest_output_buf);
	int ret;

	/* left trim the file path for readability */
	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg,
	    size,
	    "\n[KTEST]\t"                 /* header */
	    "TESTBEGIN\t"               /* type */
	    "%lld\t"                    /* time */
	    "%d\t"                      /* index */
	    "%s\t"                      /* file */
	    "%d\t"                      /* line */
	    "%s\n",                     /* name */
	    ktest_current_time,
	    ktest_test_index,
	    fname,
	    ktest_current_line,
	    test_name);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

void
ktest_emit_testskip(const char * skip_msg, va_list args)
{
	char * msg = ktest_output_buf;
	int size = sizeof(ktest_output_buf);
	int ret;

	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg,
	    size,
	    "\n[KTEST]\t"                /* header */
	    "TESTSKIP\t"               /* type */
	    "%lld\t"                   /* time */
	    "%s\t"                     /* file */
	    "%d\t",                    /* line */
	    ktest_current_time,
	    fname,
	    ktest_current_line);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = vsnprintf(msg, size, skip_msg, args);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = snprintf(msg, size, "\n");
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

void
ktest_emit_testend()
{
	char * msg = ktest_output_buf;
	int size = sizeof(ktest_output_buf);
	int ret;

	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg,
	    size,
	    "\n[KTEST]\t"                 /* header */
	    "TESTEND\t"                 /* type */
	    "%lld\t"                    /* time */
	    "%d\t"                      /* index */
	    "%s\t"                      /* file */
	    "%d\t"                      /* line */
	    "%s\n",                     /* name */
	    ktest_current_time,
	    ktest_test_index,
	    fname,
	    ktest_current_line,
	    ktest_test_name);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

void
ktest_emit_log(const char * log_msg, va_list args)
{
	char * msg = ktest_output_buf;
	int size = sizeof(ktest_output_buf);
	int ret;

	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg,
	    size,
	    "\n[KTEST]\t"            /* header */
	    "LOG\t"                /* type */
	    "%lld\t"               /* time */
	    "%s\t"                 /* file */
	    "%d\t",                /* line */
	    ktest_current_time,
	    fname,
	    ktest_current_line);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = vsnprintf(msg, size, log_msg, args);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = snprintf(msg, size, "\n");
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

void
ktest_emit_perfdata(const char * metric, const char * unit, double value, const char * desc)
{
	static const char * perfstr = "%s\t%lld\t%s\t\"%s\"";
	char * msg = ktest_output_buf;
	int64_t print_value = (int64_t)value;
	int size   = sizeof(ktest_output_buf);
	int ret;

	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg, size,
	    "\n[KTEST]\t"            /* header */
	    "PERF\t"               /* type */
	    "%lld\t"               /* time */
	    "%s\t"                 /* file */
	    "%d\t",                /* line */
	    ktest_current_time,
	    fname,
	    ktest_current_line);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = snprintf(msg, size, perfstr, metric, print_value, unit, desc);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	ret = snprintf(msg, size, "\n");
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

void
ktest_emit_testcase(void)
{
	char * msg = ktest_output_buf;
	int size = sizeof(ktest_output_buf);
	int ret;

	const char *fname = strnstr((char *)(uintptr_t)ktest_current_file, "xnu", 100);

	ret = snprintf(msg,
	    size,
	    "\n[KTEST]\t"            /* header */
	    "%s\t"                 /* type */
	    "%lld\t"               /* time */
	    "%d\t"                 /* index */
	    "%s\t"                 /* file */
	    "%d\t"                 /* line */
	    "%s\t"                 /* message */
	    "%s",                  /* current_expr */
	    ktest_testcase_result_tokens[ktest_testcase_mode]
	    [ktest_testcase_result],
	    ktest_current_time,
	    ktest_expression_index,
	    fname,
	    ktest_current_line,
	    ktest_current_msg,
	    ktest_current_expr);
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	for (int i = 0; ktest_current_var_names[i][0]; i++) {
		ret = snprintf(msg,
		    size,
		    "\t%s\t%s",
		    ktest_current_var_names[i],
		    ktest_current_var_values[i]);
		BOUNDS_CHECK_AND_UPDATE(ret, size);
	}

	ret = snprintf(msg, size, "\n");
	BOUNDS_CHECK_AND_UPDATE(ret, size);

	EMIT(ktest_output_buf, (int)(msg - ktest_output_buf));
}

