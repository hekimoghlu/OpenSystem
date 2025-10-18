module Docs
  class GjsScraper < FileScraper
    self.name = 'GJS'
    self.type = 'gjs'
    self.root_path = 'README.md'
    self.initial_paths = %w[
      ByteArray
      cairo
      Console
      Encoding
      Environment
      ESModules
      Format
      Gettext
      Lang
      Logging
      Mainloop
      Mapping
      Overrides
      Profiling
      Signals
      System
      Testing
      Timers
    ].map { |name| name + '.md' }

    html_filters.push 'gjs_scraper/entries', 'gjs_scraper/clean_html'

    options[:title] = 'GJS'
    options[:skip_links] = true

    # Transform absolute URIs into relative URIs
    options[:fix_urls] = ->(url) { url.gsub!(/^https:\/\/gjs-docs.gnome.org/, '..') }

    options[:attribution] = <<-HTML
      &copy; 2022 GJS Contributors<br>
      Licensed under the MIT License.
    HTML

    def get_latest_version(opts)
      tags = get_gitlab_tags('gitlab.gnome.org', 'GNOME', 'gjs', opts)
      tags[0]['name']
    end
  end
end
