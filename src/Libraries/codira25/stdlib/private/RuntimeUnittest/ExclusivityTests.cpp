/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "language/Runtime/Exclusivity.h"
#include "language/Runtime/Metadata.h"

using namespace language;

LANGUAGE_CC(language) LANGUAGE_RUNTIME_LIBRARY_VISIBILITY extern "C"
void testExclusivityNullPC() {
  ValueBuffer scratch, scratch2;
  long var;
  language_beginAccess(&var, &scratch,
                    ExclusivityFlags::Read | ExclusivityFlags::Tracking,
                    /*pc=*/0);
  language_beginAccess(&var, &scratch2, ExclusivityFlags::Modify,
                    /*pc=*/0);
  language_endAccess(&scratch2);
  language_endAccess(&scratch);
}

LANGUAGE_CC(language) LANGUAGE_RUNTIME_LIBRARY_VISIBILITY extern "C"
void testExclusivityPCOne() {
  ValueBuffer scratch, scratch2;
  long var;
  language_beginAccess(&var, &scratch,
                    ExclusivityFlags::Read | ExclusivityFlags::Tracking,
                    /*pc=*/(void *)1);
  language_beginAccess(&var, &scratch2, ExclusivityFlags::Modify,
                    /*pc=*/(void *)1);
  language_endAccess(&scratch2);
  language_endAccess(&scratch);
}

LANGUAGE_CC(language) LANGUAGE_RUNTIME_LIBRARY_VISIBILITY extern "C"
void testExclusivityBogusPC() {
  ValueBuffer scratch, scratch2;
  long var;
  language_beginAccess(&var, &scratch,
                    ExclusivityFlags::Read | ExclusivityFlags::Tracking,
                    /*pc=*/(void *)0xdeadbeefdeadbeefULL);
  language_beginAccess(&var, &scratch2, ExclusivityFlags::Modify,
                    /*pc=*/(void *)0xdeadbeefdeadbeefULL);
  language_endAccess(&scratch2);
  language_endAccess(&scratch);
}


// rdar://32866493
LANGUAGE_CC(language) LANGUAGE_RUNTIME_LIBRARY_VISIBILITY extern "C"
void testExclusivityNonNested() {
  const int N = 5;
  ValueBuffer scratches[N];
  long vars[N];

  auto begin = [&](unsigned i) {
    assert(i < N);
    language_beginAccess(&vars[i], &scratches[i],
                      ExclusivityFlags::Modify | ExclusivityFlags::Tracking, 0);
  };
  auto end = [&](unsigned i) {
    assert(i < N);
    language_endAccess(&scratches[i]);
    memset(&scratches[i], /*gibberish*/ 0x99, sizeof(ValueBuffer));
  };
  auto accessAll = [&] {
    for (unsigned i = 0; i != N; ++i) begin(i);
    for (unsigned i = 0; i != N; ++i) end(i);
  };

  accessAll();
  begin(0); begin(1); end(0); end(1);
  begin(0); begin(1); end(0); end(1);
  accessAll();
  begin(1); begin(0); begin(2); end(0); end(2); end(1);
  accessAll();
  begin(0); begin(1); begin(2); begin(3); begin(4);
  end(1); end(4); end(0); end(2); end(3);
  accessAll();
}
