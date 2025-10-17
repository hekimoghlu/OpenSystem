/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#ifndef _METRICS_H_
#define _METRICS_H_

#include <sys/time.h>
#include <stdio.h>

// mtree error logging
enum mtree_result {
	SUCCESS = 0,
	WARN_TIME = -1,
	WARN_USAGE = -2,
	WARN_CHECKSUM = -3,
	WARN_MISMATCH = -4,
	WARN_UNAME = -5,
	/* Could also be a POSIX errno value */
};

void set_metrics_file(FILE *file);
void set_metric_start_time(time_t time);
void set_metric_path(char *path);
#define RECORD_FAILURE(location, error) mtree_record_failure(location, error)
void mtree_record_failure(int location, int code);
void print_metrics_to_file(void);

#endif /* _METRICS_H_ */
