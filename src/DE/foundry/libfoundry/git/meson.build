if not get_option('feature-vcs')
  error('-Dfeature-git=true requires -Dfeature-vcs=true')
endif

libfoundry_config_conf.set10('FOUNDRY_FEATURE_GIT', true)

foundry_sources += files([
  'foundry-git-blame.c',
  'foundry-git-branch.c',
  'foundry-git-commit.c',
  'foundry-git-cloner.c',
  'foundry-git-diff.c',
  'foundry-git-delta.c',
  'foundry-git-file.c',
  'foundry-git-reference.c',
  'foundry-git-remote.c',
  'foundry-git-signature.c',
  'foundry-git-stats.c',
  'foundry-git-status-entry.c',
  'foundry-git-status-list.c',
  'foundry-git-tag.c',
  'foundry-git-tree.c',
  'foundry-git-uri.c',
  'foundry-git-vcs.c',
])

foundry_private_sources += files([
  'foundry-git-callbacks.c',
  'foundry-git-error.c',
  'foundry-git-file-list.c',
  'foundry-git-line-changes.c',
  'foundry-git-monitor.c',
  'foundry-git-repository.c',
  'foundry-git-time.c',
  'foundry-git.c',
  'line-cache.c',
])

foundry_headers += files([
  'foundry-git-blame.h',
  'foundry-git-branch.h',
  'foundry-git-commit.h',
  'foundry-git-cloner.h',
  'foundry-git-diff.h',
  'foundry-git-delta.h',
  'foundry-git-file.h',
  'foundry-git-reference.h',
  'foundry-git-remote.h',
  'foundry-git-signature.h',
  'foundry-git-stats.h',
  'foundry-git-status-entry.h',
  'foundry-git-status-list.h',
  'foundry-git-tag.h',
  'foundry-git-tree.h',
  'foundry-git-uri.h',
  'foundry-git-vcs.h',
])

foundry_deps += [
  dependency('libgit2', version: '>= 1.6'),
  dependency('libssh2'),
]

foundry_include_directories += [include_directories('.')]
