/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#include <stdlib.h>
#include <sys/sysctl.h>
#include <darwintest.h>
#include <perfdata/perfdata.h>

typedef uint32_t entropy_sample_t;

T_GLOBAL_META(T_META_NAMESPACE("xnu.crypto"),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("crypto"));

T_DECL(entropy_collect, "Collect entropy for offline analysis",
    T_META_REQUIRES_SYSCTL_EQ("kern.development", 1),
    T_META_BOOTARGS_SET("entropy-analysis-sample-count=1000"))
{
	int ret;
	uint32_t entropy_size = 0;
	size_t size = sizeof(entropy_size);

	ret = sysctlbyname("kern.entropy.analysis.buffer_size", &entropy_size, &size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "sysctlbyname kern.entropy.analysis.buffer_size");

	uint32_t entropy_count = entropy_size / sizeof(entropy_sample_t);
	entropy_sample_t *entropy = calloc(entropy_count, sizeof(entropy_sample_t));
	size = entropy_size;

	ret = sysctlbyname("kern.entropy.analysis.buffer", entropy, &size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "sysctlbyname kern.entropy.analysis.buffer");

	// This test is not an entropy assessment. We're just checking to
	// make sure the machinery of the entropy collection sysctl seems
	// to be working.
	for (uint32_t i = 0; i < entropy_count; i += 1) {
		T_QUIET; T_EXPECT_NE(entropy[i], 0, "entropy buffer null sample %u", i);
	}

	free(entropy);
}

T_DECL(entropy_filter_rate, "Sample entropy filter rate")
{
	int ret;
	uint64_t total_sample_count = 0;
	uint64_t rejected_sample_count = 0;
	size_t size = sizeof(total_sample_count);

	ret = sysctlbyname("kern.entropy.filter.total_sample_count", &total_sample_count, &size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "kern.entropy.filter.total_sample_count");

	size = sizeof(rejected_sample_count);
	ret = sysctlbyname("kern.entropy.filter.rejected_sample_count", &rejected_sample_count, &size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "kern.entropy.filter.rejected_sample_count");

	double rejection_rate = (double) rejected_sample_count / (double) total_sample_count;

	pdwriter_t writer = pdwriter_open_tmp("xnu", "entropy_filter_rate", 0, 0, NULL, 0);
	T_ASSERT_NOTNULL(writer, "pdwriter_open_tmp");

	pdwriter_new_value(writer, "Rejection Rate", PDUNIT_CUSTOM(rejectrate), rejection_rate);

	pdwriter_close(writer);
}
