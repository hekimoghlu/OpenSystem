/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include "SOSCircle_regressions.h"

#include "keychain/SecureObjectSync/SOSDigestVector.h"

#include <utilities/SecCFRelease.h>
#include <stdlib.h>
#include "SOSRegressionUtilities.h"

static void testNullDigestVector(void)
{
    struct SOSDigestVector dv = SOSDigestVectorInit;
    is(dv.count, (size_t)0, "count is 0");
    is(dv.capacity, (size_t)0, "capacity is 0");
    ok(!dv.unsorted, "unsorted is false");
}

static CFStringRef dvCopyString(const struct SOSDigestVector *dv)
{
    CFMutableStringRef desc = CFStringCreateMutable(kCFAllocatorDefault, 0);
    SOSDigestVectorApplySorted(dv, ^(const uint8_t *digest, bool *stop) {
        char buf[2] = {};
        buf[0] = digest[0];
        CFStringAppendCString(desc, buf, kCFStringEncodingUTF8);
    });
    return desc;
}

static void testIntersectUnionDigestVector(void)
{
    struct SOSDigestVector dv1 = SOSDigestVectorInit;
    struct SOSDigestVector dv2 = SOSDigestVectorInit;
    struct SOSDigestVector dvu = SOSDigestVectorInit;
    SOSDigestVectorAppend(&dv1, (void *)"a                  ");
    SOSDigestVectorAppend(&dv1, (void *)"b                  ");
    SOSDigestVectorAppend(&dv1, (void *)"d                  ");
    SOSDigestVectorAppend(&dv1, (void *)"f                  ");
    SOSDigestVectorAppend(&dv1, (void *)"h                  ");

    SOSDigestVectorAppend(&dv2, (void *)"c                  ");
    SOSDigestVectorAppend(&dv2, (void *)"d                  ");
    SOSDigestVectorAppend(&dv2, (void *)"e                  ");
    SOSDigestVectorAppend(&dv2, (void *)"f                  ");
    SOSDigestVectorAppend(&dv2, (void *)"g                  ");

    SOSDigestVectorAppend(&dvu, (void *)"a                  ");
    SOSDigestVectorAppend(&dvu, (void *)"b                  ");
    SOSDigestVectorAppend(&dvu, (void *)"b                  ");
    SOSDigestVectorAppend(&dvu, (void *)"b                  ");
    SOSDigestVectorAppend(&dvu, (void *)"b                  ");
    SOSDigestVectorAppend(&dvu, (void *)"c                  ");
    SOSDigestVectorAppend(&dvu, (void *)"d                  ");
    SOSDigestVectorAppend(&dvu, (void *)"f                  ");
    SOSDigestVectorAppend(&dvu, (void *)"h                  ");
    
    SOSDigestVectorAppend(&dvu, (void *)"c                  ");
    SOSDigestVectorAppend(&dvu, (void *)"d                  ");
    SOSDigestVectorAppend(&dvu, (void *)"e                  ");
    SOSDigestVectorAppend(&dvu, (void *)"f                  ");
    SOSDigestVectorAppend(&dvu, (void *)"g                  ");

    struct SOSDigestVector dvintersect = SOSDigestVectorInit;
    SOSDigestVectorIntersectSorted(&dv1, &dv2, &dvintersect);
    CFStringRef desc = dvCopyString(&dvintersect);
    ok(CFEqual(CFSTR("df"), desc), "intersection is %@", desc);
    CFReleaseNull(desc);

    struct SOSDigestVector dvunion = SOSDigestVectorInit;
    SOSDigestVectorUnionSorted(&dv1, &dv2, &dvunion);
    desc = dvCopyString(&dvunion);
    ok(CFEqual(CFSTR("abcdefgh"), desc), "union is %@", desc);
    CFReleaseNull(desc);

    struct SOSDigestVector dvdels = SOSDigestVectorInit;
    struct SOSDigestVector dvadds = SOSDigestVectorInit;
    SOSDigestVectorDiffSorted(&dv1, &dv2, &dvdels, &dvadds);
    desc = dvCopyString(&dvdels);
    ok(CFEqual(CFSTR("abh"), desc), "dels is %@", desc);
    CFReleaseNull(desc);
    desc = dvCopyString(&dvadds);
    ok(CFEqual(CFSTR("ceg"), desc), "adds is %@", desc);
    CFReleaseNull(desc);

    CFErrorRef localError = NULL;
    struct SOSDigestVector dvpatched = SOSDigestVectorInit;
    ok(SOSDigestVectorPatch(&dv1, &dvdels, &dvadds, &dvpatched, &localError), "patch : %@", localError);
    CFReleaseNull(localError);
    desc = dvCopyString(&dvpatched);
    ok(CFEqual(CFSTR("cdefg"), desc), "patched dv1 - dels + adds is %@, should be: %@", desc, CFSTR("cdefg"));
    CFReleaseNull(desc);

    SOSDigestVectorFree(&dvpatched);
    ok(SOSDigestVectorPatch(&dv2, &dvadds, &dvdels, &dvpatched, &localError), "patch : %@", localError);
    CFReleaseNull(localError);
    desc = dvCopyString(&dvpatched);
    ok(CFEqual(CFSTR("abdfh"), desc), "patched dv2 - adds + dels is is %@, should be: %@", desc, CFSTR("abdfh"));
    CFReleaseNull(desc);

    SOSDigestVectorAppend(&dvadds, (void *)"c                  ");
    SOSDigestVectorFree(&dvpatched);
    SOSDigestVectorUniqueSorted(&dvadds);
    ok(SOSDigestVectorPatch(&dv2, &dvadds, &dvdels, &dvpatched, &localError), "patch failed: %@", localError);
    CFReleaseNull(localError);
    desc = dvCopyString(&dvpatched);
    ok(CFEqual(CFSTR("abdfh"), desc), "patched dv2 - adds + dels is is %@, should be: %@", desc, CFSTR("abdfh"));
    CFReleaseNull(desc);
    
    SOSDigestVectorUniqueSorted(&dvu);
    desc = dvCopyString(&dvu);
    ok(CFEqual(CFSTR("abcdefgh"), desc), "uniqued dvu is %@, should be: %@", desc, CFSTR("abcdefgh"));
    CFReleaseNull(desc);

    // This operation should be idempotent
    SOSDigestVectorUniqueSorted(&dvu);
    desc = dvCopyString(&dvu);
    ok(CFEqual(CFSTR("abcdefgh"), desc), "uniqued dvu is %@, should be: %@", desc, CFSTR("abcdefgh"));
    CFReleaseNull(desc);

    SOSDigestVectorFree(&dv1);
    SOSDigestVectorFree(&dv2);
    SOSDigestVectorFree(&dvu);
    SOSDigestVectorFree(&dvintersect);
    SOSDigestVectorFree(&dvunion);
    SOSDigestVectorFree(&dvdels);
    SOSDigestVectorFree(&dvadds);
    SOSDigestVectorFree(&dvpatched);
}

static void tests(void)
{
    testNullDigestVector();
    testIntersectUnionDigestVector();
}

int sc_45_digestvector(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(15);
    tests();
#else
    plan_tests(0);
#endif
	return 0;
}
