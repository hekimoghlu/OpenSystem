/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
struct kinfo;
struct nlist;
struct var;
struct varent;

extern fixpt_t ccpu;
extern int cflag, eval, fscale, nlistread, rawcpu;
#ifdef __APPLE__
extern uint64_t mempages;
#else
extern unsigned long mempages;
#endif
extern time_t now;
extern int showthreads, sumrusage, termwidth, totwidth;
extern STAILQ_HEAD(velisthead, varent) varlist;

__BEGIN_DECLS
int	 get_task_info(KINFO *);
void	 command(KINFO *, VARENT *);
void	 just_command(KINFO *, VARENT *);
void	 args(KINFO *, VARENT *);
int	 s_command(KINFO *);
int	 s_just_command(KINFO *);
int	 s_args(KINFO *);
void	 cputime(KINFO *, VARENT *);
void	 pstime(KINFO *, VARENT *);
void	 p_etime(KINFO *, VARENT *);
int	 s_etime(KINFO *);
void	 putime(KINFO *, VARENT *);
int	 donlist(void);
void	 evar(KINFO *, VARENT *);
VARENT	*find_varentry(VAR *);
const	 char *fmt_argv(char **, char *, size_t);
int	 getpcpu(KINFO *);
double	 getpmem(KINFO *);
void	 logname(KINFO *, VARENT *);
void	 longtname(KINFO *, VARENT *);
void	 lstarted(KINFO *, VARENT *);
void	 maxrss(KINFO *, VARENT *);
void	 nlisterr(struct nlist *);
void	 p_rssize(KINFO *, VARENT *);
void	 pagein(KINFO *, VARENT *);
void	 parsefmt(const char *, int);
#ifdef __APPLE__
void	 persona(KINFO *, VARENT *);
#endif /* __APPLE__ */
void	 pcpu(KINFO *, VARENT *);
void	 pmem(KINFO *, VARENT *);
void	 pri(KINFO *, VARENT *);
void	 rtprior(KINFO *, VARENT *);
void	 printheader(void);
void	 pvar(KINFO *, VARENT *);
void	 runame(KINFO *, VARENT *);
void	 rvar(KINFO *, VARENT *);
int	 s_runame(KINFO *);
int	 s_uname(KINFO *);
void	 showkey(void);
void	 started(KINFO *, VARENT *);
void	 state(KINFO *, VARENT *);
void	 tdev(KINFO *, VARENT *);
void	 tname(KINFO *, VARENT *);
void	 tsize(KINFO *, VARENT *);
void	 ucomm(KINFO *, VARENT *);
void	 uname(KINFO *, VARENT *);
void	 uvar(KINFO *, VARENT *);
void	 vsize(KINFO *, VARENT *);
void	 wchan(KINFO *, VARENT *);
void	 wq(KINFO *, VARENT *);
__END_DECLS
