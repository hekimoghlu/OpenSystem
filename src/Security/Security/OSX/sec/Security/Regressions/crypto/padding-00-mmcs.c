/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#include "test/testmore.h"

#include <Security/SecPaddingConfigurationsPriv.h>
#include "shared_regressions.h"
#include <utilities/SecCFWrappers.h>

#define kTestTestCount 67

struct test_vector {
	SecPaddingType type;
	uint32_t originalSize;
	int64_t paddedSize;
};
typedef struct test_vector test_vector_t;

test_vector_t test_vectors[] = {
	{SecPaddingTypeMMCS,0,64},
	{SecPaddingTypeMMCS,1,64},
	{SecPaddingTypeMMCS,2,64},
	{SecPaddingTypeMMCS,64,64},
	{SecPaddingTypeMMCS,65,128},
	{SecPaddingTypeMMCS,68,128},
	{SecPaddingTypeMMCS,80,128},
	{SecPaddingTypeMMCS,90,128},
	{SecPaddingTypeMMCS,100,128},
	{SecPaddingTypeMMCS,128,128},
	{SecPaddingTypeMMCS,129,256},
	{SecPaddingTypeMMCS,129,256},
	{SecPaddingTypeMMCS,150,256},
	{SecPaddingTypeMMCS,256,256},
	{SecPaddingTypeMMCS,257,512},
	{SecPaddingTypeMMCS,512,512},
	{SecPaddingTypeMMCS,612,1024},
	{SecPaddingTypeMMCS,1000,1024},
	{SecPaddingTypeMMCS,1024,1024},
	{SecPaddingTypeMMCS,1025,2048},
	{SecPaddingTypeMMCS,1800,2048},
	{SecPaddingTypeMMCS,2000,2048},
	{SecPaddingTypeMMCS,2047,2048},
	{SecPaddingTypeMMCS,2048,2048},
	{SecPaddingTypeMMCS,3072,3072},
	{SecPaddingTypeMMCS,4096,4096},
	{SecPaddingTypeMMCS,31744,31744},
	{SecPaddingTypeMMCS,31999,32768},
	{SecPaddingTypeMMCS,32000,32768},
	{SecPaddingTypeMMCS,32001,32768},
	{SecPaddingTypeMMCS,32769,40960},
	{SecPaddingTypeMMCS,4294967295,4294967296}
};

static void tests_negative(void)
{
	CFErrorRef testError = NULL;

	ok(SecPaddingCompute(SecPaddingTypeMMCS, 0, &testError) == 64 && testError==NULL);

	// Wrong type
	ok(SecPaddingCompute(100, 20, &testError) == -1);
	ok(testError!=NULL);
	CFReleaseNull(testError);
}

static void tests_positive(void)
{
	CFErrorRef testError = NULL;
	size_t n;

	n = sizeof(test_vectors)/sizeof(test_vectors[0]);
	for(size_t i=0; i<n; i++) {
		is((SecPaddingCompute(test_vectors[i].type, test_vectors[i].originalSize, &testError) + test_vectors[i].originalSize),
		   test_vectors[i].paddedSize, "Test # %zu, type %ld: input %u",i,(long)test_vectors[i].type,
		   test_vectors[i].originalSize);
		ok(testError==NULL);
	}
}

int padding_00_mmcs(int argc, char *const *argv)
{
	plan_tests(kTestTestCount);

	tests_positive();
	tests_negative();

	return 0;
}

