/*
 * ggit-cred-ssh-interactive.h
 * This file is part of libgit2-glib
 *
 * Copyright (C) 2014 - Jesse van den Kieboom
 *
 * libgit2-glib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libgit2-glib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libgit2-glib. If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef __GGIT_CRED_SSH_INTERACTIVE_H__
#define __GGIT_CRED_SSH_INTERACTIVE_H__

#include <glib-object.h>
#include <libgit2-glib/ggit-cred.h>

G_BEGIN_DECLS

#define GGIT_TYPE_CRED_SSH_INTERACTIVE (ggit_cred_ssh_interactive_get_type ())
G_DECLARE_DERIVABLE_TYPE (GgitCredSshInteractive, ggit_cred_ssh_interactive, GGIT, CRED_SSH_INTERACTIVE, GgitCred)

#define GGIT_TYPE_CRED_SSH_INTERACTIVE_PROMPT	(ggit_cred_ssh_interactive_prompt_get_type ())
#define GGIT_CRED_SSH_INTERACTIVE_PROMPT(obj)	((GgitCredSshInteractivePrompt*)obj)

struct _GgitCredSshInteractiveClass
{
	/*< private >*/
	GgitCredClass parent_class;

	/* virtual methods */
	void (*prompt) (GgitCredSshInteractive        *cred,
	                GgitCredSshInteractivePrompt **prompts,
	                gsize                          num_prompts);
};

GgitCredSshInteractive *
              ggit_cred_ssh_interactive_new           (const gchar             *username,
                                                       GError                 **error);

const gchar  *ggit_cred_ssh_interactive_get_username  (GgitCredSshInteractive  *cred);


GType         ggit_cred_ssh_interactive_prompt_get_type
                                               (void) G_GNUC_CONST;

GgitCredSshInteractivePrompt *
              ggit_cred_ssh_interactive_prompt_new
                                               (const gchar                  *name,
                                                const gchar                  *instruction,
                                                const gchar                  *text,
                                                gboolean                      is_masked);

GgitCredSshInteractivePrompt *
              ggit_cred_ssh_interactive_prompt_ref
                                               (GgitCredSshInteractivePrompt *prompt);

void          ggit_cred_ssh_interactive_prompt_unref
                                               (GgitCredSshInteractivePrompt *prompt);

const gchar  *ggit_cred_ssh_interactive_prompt_get_name
                                               (GgitCredSshInteractivePrompt *prompt);

const gchar  *ggit_cred_ssh_interactive_prompt_get_text
                                               (GgitCredSshInteractivePrompt *prompt);

const gchar  *ggit_cred_ssh_interactive_prompt_get_instruction
                                               (GgitCredSshInteractivePrompt *prompt);

gboolean      ggit_cred_ssh_interactive_prompt_is_masked
                                               (GgitCredSshInteractivePrompt *prompt);

void          ggit_cred_ssh_interactive_prompt_set_response
                                               (GgitCredSshInteractivePrompt *prompt,
                                                const gchar                  *response);

const gchar  *ggit_cred_ssh_interactive_prompt_get_response
                                               (GgitCredSshInteractivePrompt *prompt);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GgitCredSshInteractivePrompt, ggit_cred_ssh_interactive_prompt_unref)

G_END_DECLS

#endif /* __GGIT_CRED_SSH_INTERACTIVE_H__ */

/* ex:set ts=8 noet: */
