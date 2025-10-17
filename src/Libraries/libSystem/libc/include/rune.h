/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#ifndef	_RUNE_H_
#define	_RUNE_H_

#include <runetype.h>
#include <stdio.h>
#include <Availability.h>

/*--------------------------- DEPRECIATED -------------------------------
 * This interface is depreciated and will eventually be removed.  The ISO C99
 * extended multibyte and wide character facilities should be used instead.
 * See multibyte(3) and related man pages for further details.
 *--------------------------- DEPRECIATED -------------------------------*/

#define	_PATH_LOCALE	"/usr/share/locale"

#define _INVALID_RUNE   _CurrentRuneLocale->__invalid_rune

#define __sgetrune      _CurrentRuneLocale->__sgetrune
#define __sputrune      _CurrentRuneLocale->__sputrune

#define sgetrune(s, n, r)       (*__sgetrune)((s), (n), (r))
#define sputrune(c, s, n, r)    (*__sputrune)((c), (s), (n), (r))

__BEGIN_DECLS
char	*mbrune(const char *, rune_t) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
char	*mbrrune(const char *, rune_t) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
char	*mbmb(const char *, char *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
long	 fgetrune(FILE *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
int	 fputrune(rune_t, FILE *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
int	 fungetrune(rune_t, FILE *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
int	 setrunelocale(char *) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
void	 setinvalidrune(rune_t) __OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_0, __MAC_10_4, __IPHONE_NA, __IPHONE_NA);
__END_DECLS

#endif	/*! _RUNE_H_ */
