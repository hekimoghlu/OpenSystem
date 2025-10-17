/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
/*
 * This short file provides a home for the stack of saved contexts.
 * The actions for saving and restoring are encapsulated within
 * individual modules.
 */

#include "zsh.mdh"
#include "context.pro"

struct context_stack {
    struct context_stack *next;

    struct hist_stack hist_stack;
    struct lex_stack lex_stack;
    struct parse_stack parse_stack;
};

static struct context_stack *cstack;

/* save some or all of current context */

/**/
mod_export void
zcontext_save_partial(int parts)
{
    struct context_stack *cs;

    queue_signals();

    cs = (struct context_stack *)malloc(sizeof(struct context_stack));

    if (parts & ZCONTEXT_HIST) {
	hist_context_save(&cs->hist_stack, !cstack);
    }
    if (parts & ZCONTEXT_LEX) {
	lex_context_save(&cs->lex_stack, !cstack);
    }
    if (parts & ZCONTEXT_PARSE) {
	parse_context_save(&cs->parse_stack, !cstack);
    }

    cs->next = cstack;
    cstack = cs;

    unqueue_signals();
}

/* save context in full */

/**/
mod_export void
zcontext_save(void)
{
    zcontext_save_partial(ZCONTEXT_HIST|ZCONTEXT_LEX|ZCONTEXT_PARSE);
}

/* restore context or part thereof */

/**/
mod_export void
zcontext_restore_partial(int parts)
{
    struct context_stack *cs = cstack;

    DPUTS(!cstack, "BUG: zcontext_restore() without zcontext_save()");

    queue_signals();
    cstack = cstack->next;

    if (parts & ZCONTEXT_HIST) {
	hist_context_restore(&cs->hist_stack, !cstack);
    }
    if (parts & ZCONTEXT_LEX) {
	lex_context_restore(&cs->lex_stack, !cstack);
    }
    if (parts & ZCONTEXT_PARSE) {
	parse_context_restore(&cs->parse_stack, !cstack);
    }

    free(cs);

    unqueue_signals();
}

/* restore full context */

/**/
mod_export void
zcontext_restore(void)
{
    zcontext_restore_partial(ZCONTEXT_HIST|ZCONTEXT_LEX|ZCONTEXT_PARSE);
}
