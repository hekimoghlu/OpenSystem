/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include <sys/types.h>
#include <sys/param.h>
#include <db.h>

/* structures */

/* All times are stored in 1e-6s units. */

struct cmdinfo {
	char		ci_comm[MAXCOMLEN+2];	/* command name (+ '*') */
	uid_t		ci_uid;			/* user id */
	u_quad_t	ci_calls;		/* number of calls */
#ifdef __APPLE__
	u_quad_t	ci_etime;		/* elapsed time */
	u_quad_t	ci_utime;		/* user time */
	u_quad_t	ci_stime;		/* system time */
	u_quad_t	ci_mem;			/* memory use */
	u_quad_t	ci_io;			/* number of disk i/o ops */
#else
	double		ci_etime;		/* elapsed time */
	double		ci_utime;		/* user time */
	double		ci_stime;		/* system time */
	double		ci_mem;			/* memory use */
	double		ci_io;			/* number of disk i/o ops */
#endif
	u_int		ci_flags;		/* flags; see below */
};
#define	CI_UNPRINTABLE	0x0001			/* unprintable chars in name */

struct userinfo {
	uid_t		ui_uid;			/* user id; for consistency */
	u_quad_t	ui_calls;		/* number of invocations */
#ifdef __APPLE__
	u_quad_t	ui_utime;		/* user time */
	u_quad_t	ui_stime;		/* system time */
	u_quad_t	ui_mem;			/* memory use */
	u_quad_t	ui_io;			/* number of disk i/o ops */
#else
	double		ui_utime;		/* user time */
	double		ui_stime;		/* system time */
	double		ui_mem;			/* memory use */
	double		ui_io;			/* number of disk i/o ops */
#endif
};

/* typedefs */

typedef	int (*cmpf_t)(const DBT *, const DBT *);

/* external functions in db.c */
int db_copy_in(DB **mdb, const char *dbname, const char *name,
    BTREEINFO *bti, int (*v1_to_v2)(DBT *key, DBT *data));
int db_copy_out(DB *mdb, const char *dbname, const char *name,
    BTREEINFO *bti);
void db_destroy(DB *db, const char *uname);

/* external functions in pdb.c */
int	pacct_init(void);
void	pacct_destroy(void);
int	pacct_add(const struct cmdinfo *);
int	pacct_update(void);
void	pacct_print(void);

#ifndef __APPLE__
/* external functions in readrec.c */
int	readrec_forward(FILE *f, struct acctv3 *av2);
#endif

/* external functions in usrdb.c */
int	usracct_init(void);
void	usracct_destroy(void);
int	usracct_add(const struct cmdinfo *);
int	usracct_update(void);
void	usracct_print(void);

/* variables */

extern int	aflag, bflag, cflag, dflag, Dflag, fflag, iflag, jflag, kflag;
extern int	Kflag, lflag, mflag, qflag, rflag, sflag, tflag, uflag, vflag;
extern u_quad_t	cutoff;
extern cmpf_t	sa_cmp;
extern const char *pdb_file, *usrdb_file;

/* some #defines to help with db's stupidity */

#define	DB_CLOSE(db) \
	((*(db)->close)(db))
#define	DB_GET(db, key, data, flags) \
	((*(db)->get)((db), (key), (data), (flags)))
#define	DB_PUT(db, key, data, flags) \
	((*(db)->put)((db), (key), (data), (flags)))
#define	DB_SYNC(db, flags) \
	((*(db)->sync)((db), (flags)))
#define	DB_SEQ(db, key, data, flags) \
	((*(db)->seq)((db), (key), (data), (flags)))
