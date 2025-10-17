/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include "db_config.h"

#include "db_int.h"
#include "dbinc/log.h"

/*
 * log_compare --
 *	Compare two LSN's; return 1, 0, -1 if first is >, == or < second.
 *
 * EXTERN: int log_compare __P((const DB_LSN *, const DB_LSN *));
 */
int
log_compare(lsn0, lsn1)
	const DB_LSN *lsn0, *lsn1;
{
	return (LOG_COMPARE(lsn0, lsn1));
}

/*
 * __log_check_page_lsn --
 *	Panic if the page's lsn in past the end of the current log.
 *
 * PUBLIC: int __log_check_page_lsn __P((ENV *, DB *, DB_LSN *));
 */
int
__log_check_page_lsn(env, dbp, lsnp)
	ENV *env;
	DB *dbp;
	DB_LSN *lsnp;
{
	LOG *lp;
	int ret;

	lp = env->lg_handle->reginfo.primary;
	LOG_SYSTEM_LOCK(env);

	ret = LOG_COMPARE(lsnp, &lp->lsn);

	LOG_SYSTEM_UNLOCK(env);

	if (ret < 0)
		return (0);

	__db_errx(env,
	    "file %s has LSN %lu/%lu, past end of log at %lu/%lu",
	    dbp == NULL || dbp->fname == NULL ? "unknown" : dbp->fname,
	    (u_long)lsnp->file, (u_long)lsnp->offset,
	    (u_long)lp->lsn.file, (u_long)lp->lsn.offset);
	__db_errx(env, "%s",
    "Commonly caused by moving a database from one database environment");
	__db_errx(env, "%s",
    "to another without clearing the database LSNs, or by removing all of");
	__db_errx(env, "%s",
    "the log files from a database environment");
	return (EINVAL);
}
