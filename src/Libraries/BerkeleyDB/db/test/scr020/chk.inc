/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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

#!/bin/sh -
#
# $Id: chk.inc,v 12.1 2008/05/07 12:46:43 bschmeck Exp $
#
# Check for inclusion of db_config.h after "const" or other includes.

d=../..

# Test must be run from the top-level directory, not from a test directory.
[ -f $d/LICENSE ] || {
	echo 'FAIL: cannot find source distribution directory.'
	exit 1
}

t1=__1
t2=__2

(cd $d && find . -name '*.[chys]' -o -name '*.cpp' |
    xargs egrep -l '#include.*db_config.h') | tee /tmp/o |
    sed -e '/dbdemo.c$/d' \
        -e '/db_java_wrap.c$/d' \
	-e '/ex_apprec.c$/d' > $t1

(for i in `cat $t1`; do
	egrep -w 'db_config.h|const' /dev/null $d/$i | head -1
done) > $t2

if egrep const $t2 > /dev/null; then
	echo 'FAIL: found const before include of db_config.h'
	egrep const $t2
	exit 1
fi

:> $t2
for i in `cat $t1`; do
	egrep -w '#include' /dev/null $d/$i | head -1 >> $t2
done

if egrep -v db_config.h $t2 > /dev/null; then
	echo 'FAIL: found includes before include of db_config.h'
	egrep -v db_config.h $t2
	exit 1
fi

exit 0
