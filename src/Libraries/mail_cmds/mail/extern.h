/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
struct name *cat(struct name *, struct name *);
struct name *delname(struct name *, char *);
struct name *elide(struct name *);
struct name *extract(char *, int);
struct name *gexpand(struct name *, struct grouphead *, int, int);
struct name *nalloc(char *, int);
struct name *outof(struct name *, FILE *, struct header *);
struct name *put(struct name *, struct name *);
struct name *tailof(struct name *);
struct name *usermap(struct name *);
FILE	*Fdopen(int, const char *);
FILE	*Fopen(const char *, const char *);
FILE	*Popen(char *, const char *);
FILE	*collect(struct header *, int);
char	*copyin(char *, char **);
char	*detract(struct name *, int);
char	*expand(char *);
char	*getauthor(char *);
char	*getdeadletter(void);
char	*getname(uid_t);
char	*hfield(const char *, struct message *);
FILE	*infix(struct header *, FILE *);
char	*ishfield(char *, char *, const char *);
char	*name1(struct message *, int);
char	*nameof(struct message *, int);
char	*nextword(char *, char *);
char	*readtty(const char *, char *);
char 	*reedit(char *);
FILE	*run_editor(FILE *, off_t, int, int);
char	*salloc(int);
char	*savestr(char *);
FILE	*setinput(struct message *);
char	*skin(char *);
char	*skip_comment(char *);
char	*snarf(char [], int *, int);
char	*username(void);
char	*value(const char *);
char	*vcopy(const char *);
char	*yankword(char *, char *);
char	*yanklogin(char *, char *);
int	 Fclose(FILE *);
int	 More(void *);
int	 Pclose(FILE *);
int	 Respond(int *);
int	 Type(void *);
int	 doRespond(int *);
int	 dorespond(int *);
void	 alter(char *);
int	 alternates(char **);
void	 announce(void);
int	 append(struct message *, FILE *);
int	 argcount(char **);
void	 assign(const char *, const char *);
int	 bangexp(char *, size_t);
void	 brokpipe(int);
int	 charcount(char *, int);
int	 check(int, int);
void	 clob1(int);
int	 clobber(char **);
void	 close_all_files(void);
int	 cmatch(char *, char *);
void	 collhup(int);
void	 collint(int);
void	 collstop(int);
void	 commands(void);
int	 copycmd(void *v);
int	 Capcopycmd(char []);
int	 core(void);
int	 count(struct name *);
int	 deletecmd(void *);
int	 delm(int *);
int	 deltype(void *);
void	 demail(void);
int	 diction(const void *, const void *);
int	 dosh(char *);
int	 echo(char **);
int	 edit1(int *, int);
int	 editor(int *);
void	 edstop(void);
int	 elsecmd(void);
int	 endifcmd(void);
int	 evalcol(int);
int	 execute(char *, int);
int	 exwrite(char *, FILE *, int);
void	 fail(const char *, const char *);
int	 file(char **);
struct grouphead *
	 findgroup(char *);
void	 findmail(char *, char *, int);
int	 first(int, int);
void	 fixhead(struct header *, struct name *);
void	 fmt(const char *, struct name *, FILE *, int);
int	 folders(void);
int	 followup(int *);
int	 Capfollowup(int *);
int	 forward(char *, FILE *, char *, int);
void	 free_child(int);
int	 from(void *);
off_t	 fsize(FILE *);
int	 getfold(char *, int);
int	 gethfield(FILE *, char *, int, char **);
int	 getmsglist(char *, int *, int);
int	 getrawlist(char *, char **, int);
uid_t	 getuserid(char *);
int	 grabh(struct header *, int);
int	 group(void *);
void	 hangup(int);
int	 hash(const char *);
void	 hdrstop(int);
int	 headers(void *);
int	 help(void);
void	 holdsigs(void);
int	 ifcmd(char **);
int	 igcomp(const void *, const void *);
int	 igfield(void *);
int	 ignore1(char **, struct ignoretab *, const char *);
int	 igshow(struct ignoretab *, const char *);
int	 inc(void *);
int	 incfile(void);
void	 intr(int);
int	 isdate(char *);
int	 isdir(char *);
int	 isfileaddr(char *);
int	 ishead(char *);
int	 isign(const char *, struct ignoretab [2]);
int	 isprefix(const char *, const char *);
void	 istrncpy(char *, const char *, size_t);
const struct cmd *
	 lex(char *);
void	 load(char *);
struct var *
	 lookup(const char *);
int	 mail(struct name *,
	    struct name *, struct name *, struct name *, char *, char *);
void	 mail1(struct header *, int);
int	 mailpipe(char []);
void	 makemessage(FILE *, int);
void	 mark(int);
int	 markall(char *, int);
int	 matchsender(char *, int);
int	 matchfield(char *, int);
int	 mboxit(void *);
int	 member(char *, struct ignoretab *);
void	 mesedit(FILE *, int);
void	 mespipe(FILE *, char *);
int	 messize(void *);
int	 metamess(int, int);
int	 more(void *);
int	 newfileinfo(int);
int	 next(void *);
int	 null(int);
void	 parse(char *, struct headline *, char *);
int	 pcmdlist(void);
int	 pdot(void);
void	 prepare_child(sigset_t *, int, int);
int	 preserve(void *);
void	 prettyprint(struct name *);
void	 printgroup(char *);
void	 printhead(int);
int	 puthead(struct header *, FILE *, int);
int	 putline(FILE *, char *, int);
int	 pversion(int);
void	 quit(void);
int	 quitcmd(void);
int	 readline(FILE *, char *, int);
void	 register_file(FILE *, int, int);
void	 regret(int);
void	 relsesigs(void);
int	 respond(void *);
int	 retfield(void *);
int	 rexit(void *);
int	 rm(char *);
int	 run_command(char *, sigset_t *, int, int, ...);
int	 save(void *v);
int	 Capsave(char []);
int	 save1(char *, int, const char *, struct ignoretab *);
void	 savedeadletter(FILE *);
int	 saveigfield(void *);
int	 savemail(char *, FILE *);
int	 saveretfield(void *);
int	 scan(char **);
void	 scaninit(void);
int	 schdir(void *);
int	 screensize(void);
int	 scroll(void *);
int	 sendmessage(struct message *, FILE *, struct ignoretab *, char *);
int	 sendmail(char *);
int	 set(void *);
int	 setfile(char *);
void	 setmsize(int);
void	 setptr(FILE *, off_t);
void	 setscreensize(void);
int	 shell(char *);
void	 sigchild(int);
void	 sort(char **);
int	 source(char **);
void	 spreserve(void);
void	 sreset(void);
int	 start_command(char *, sigset_t *, int, int, ...);
void	 statusput(struct message *, FILE *, char *);
void	 stop(int);
int	 stouch(void *);
int	 swrite(void *);
void	 tinit(void);
int	 top(void *);
void	 touch(struct message *);
void	 ttyint(int);
void	 ttystop(int);
int	 type(void *);
int	 type1(int *, int, int);
int	 undeletecmd(void *);
void	 unmark(int);
char	**unpack(struct name *);
int	 unread(void *);
void	 unregister_file(FILE *);
int	 unset(void *);
int	 unstack(void);
void	 vfree(char *);
int	 visual(int *);
int	 wait_child(int);
int	 wait_command(int);
int	 writeback(FILE *);

extern char *__progname;
extern char *tmpdir;
