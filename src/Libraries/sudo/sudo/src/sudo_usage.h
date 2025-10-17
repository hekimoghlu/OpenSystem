/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef SUDO_USAGE_H
#define SUDO_USAGE_H

/*
 * Usage strings for sudo.  These are here because we
 * need to be able to substitute values from configure.
 */
#define SUDO_USAGE0 " -h | -V"
#define SUDO_USAGE1 " -h | -K | -k | -V"
#define SUDO_USAGE2 " -v [-ABkNnS] [-g group] [-h host] [-p prompt] [-u user]"
#define SUDO_USAGE3 " -l [-ABkNnS] [-g group] [-h host] [-p prompt] [-U user] [-u user] [command [arg ...]]"
#define SUDO_USAGE4 " [-ABbEHkNnPS] [-C num] [-D directory] [-g group] [-h host] [-p prompt] [-R directory] [-T timeout] [-u user] [VAR=value] [-i | -s] [command [arg ...]]"
#define SUDO_USAGE5 " -e [-ABkNnS] [-C num] [-D directory] [-g group] [-h host] [-p prompt] [-R directory] [-T timeout] [-u user] file ..."

/*
 * Configure script arguments used to build sudo.
 */
#define CONFIGURE_ARGS "--with-password-timeout=0 --disable-setreuid --with-env-editor --with-pam --with-libraries=bsm --with-noexec=no --sysconfdir=/private/etc --without-lecture --enable-static-sudoers --with-rundir=/var/db/sudo"

#endif /* SUDO_USAGE_H */
