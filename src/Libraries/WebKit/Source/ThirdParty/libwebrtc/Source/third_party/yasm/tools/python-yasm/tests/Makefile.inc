/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

EXTRA_DIST += tools/python-yasm/tests/python_test.sh
EXTRA_DIST += tools/python-yasm/tests/__init__.py
EXTRA_DIST += tools/python-yasm/tests/test_bytecode.py
EXTRA_DIST += tools/python-yasm/tests/test_expr.py
EXTRA_DIST += tools/python-yasm/tests/test_intnum.py
EXTRA_DIST += tools/python-yasm/tests/test_symrec.py

if HAVE_PYTHON_BINDINGS

TESTS_ENVIRONMENT += PYTHON=${PYTHON}
TESTS += tools/python-yasm/tests/python_test.sh

endif
