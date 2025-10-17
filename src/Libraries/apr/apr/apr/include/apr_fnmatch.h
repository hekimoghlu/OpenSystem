/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
/* This file has been modified by the Apache Software Foundation. */
#ifndef	_APR_FNMATCH_H_
#define	_APR_FNMATCH_H_

/**
 * @file apr_fnmatch.h
 * @brief APR FNMatch Functions
 */

#include "apr_errno.h"
#include "apr_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup apr_fnmatch Filename Matching Functions
 * @ingroup APR 
 * @{
 */

#define APR_FNM_NOMATCH     1     /**< Match failed. */
 
#define APR_FNM_NOESCAPE    0x01  /**< Disable backslash escaping. */
#define APR_FNM_PATHNAME    0x02  /**< Slash must be matched by slash. */
#define APR_FNM_PERIOD      0x04  /**< Period must be matched by period. */
#define APR_FNM_CASE_BLIND  0x08  /**< Compare characters case-insensitively. */

/**
 * Try to match the string to the given pattern, return APR_SUCCESS if
 *    match, else return APR_FNM_NOMATCH.  Note that there is no such thing as
 *    an illegal pattern.
 *
 * With all flags unset, a pattern is interpreted as such:
 *
 * PATTERN: Backslash followed by any character, including another
 *          backslash.<br/>
 * MATCHES: That character exactly.
 * 
 * <p>
 * PATTERN: ?<br/>
 * MATCHES: Any single character.
 * </p>
 * 
 * <p>
 * PATTERN: *<br/>
 * MATCHES: Any sequence of zero or more characters. (Note that multiple
 *          *s in a row are equivalent to one.)
 * 
 * PATTERN: Any character other than \?*[ or a \ at the end of the pattern<br/>
 * MATCHES: That character exactly. (Case sensitive.)
 * 
 * PATTERN: [ followed by a class description followed by ]<br/>
 * MATCHES: A single character described by the class description.
 *          (Never matches, if the class description reaches until the
 *          end of the string without a ].) If the first character of
 *          the class description is ^ or !, the sense of the description
 *          is reversed.  The rest of the class description is a list of
 *          single characters or pairs of characters separated by -. Any
 *          of those characters can have a backslash in front of them,
 *          which is ignored; this lets you use the characters ] and -
 *          in the character class, as well as ^ and ! at the
 *          beginning.  The pattern matches a single character if it
 *          is one of the listed characters or falls into one of the
 *          listed ranges (inclusive, case sensitive).  Ranges with
 *          the first character larger than the second are legal but
 *          never match. Edge cases: [] never matches, and [^] and [!]
 *          always match without consuming a character.
 * 
 * Note that these patterns attempt to match the entire string, not
 * just find a substring matching the pattern.
 *
 * @param pattern The pattern to match to
 * @param strings The string we are trying to match
 * @param flags flags to use in the match.  Bitwise OR of:
 * <pre>
 *              APR_FNM_NOESCAPE       Disable backslash escaping
 *              APR_FNM_PATHNAME       Slash must be matched by slash
 *              APR_FNM_PERIOD         Period must be matched by period
 *              APR_FNM_CASE_BLIND     Compare characters case-insensitively.
 * </pre>
 */

APR_DECLARE(apr_status_t) apr_fnmatch(const char *pattern, 
                                      const char *strings, int flags);

/**
 * Determine if the given pattern is a regular expression.
 * @param pattern The pattern to search for glob characters.
 * @return non-zero if pattern has any glob characters in it
 */
APR_DECLARE(int) apr_fnmatch_test(const char *pattern);

/**
 * Find all files that match a specified pattern in a directory.
 * @param dir_pattern The pattern to use for finding files, appended
 * to the search directory.  The pattern is anything following the
 * final forward or backward slash in the parameter.  If no slash
 * is found, the current directory is searched.
 * @param result Array to use when storing the results
 * @param p The pool to use.
 * @return APR_SUCCESS if no processing errors occurred, APR error
 * code otherwise
 * @remark The returned array may be empty even if APR_SUCCESS was
 * returned.
 */
APR_DECLARE(apr_status_t) apr_match_glob(const char *dir_pattern, 
                                         apr_array_header_t **result,
                                         apr_pool_t *p);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* !_APR_FNMATCH_H_ */
