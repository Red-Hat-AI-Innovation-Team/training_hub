/**
 * training-hub plugin for OpenCode.ai
 *
 * Registers skills directory and injects bootstrap context.
 * Tells the agent where to find training-hub scripts.
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const TrainingHubPlugin = async ({ client, directory }) => {
  const pluginRoot = path.resolve(__dirname, '../..');
  const skillsDir = path.join(pluginRoot, 'skills');
  const scriptsDir = path.join(pluginRoot, 'scripts');

  const getBootstrapContent = () => {
    return `<training-hub-plugin>
You have the training-hub LLM training plugin installed.

**Available skills:**
- training-guide — fine-tune or train language models
- setup-guide — first-time configuration

**Script paths (use these instead of \${CLAUDE_PLUGIN_ROOT}):**
- Detection: ${scriptsDir}/th_detect.sh
- Training: ${scriptsDir}/th_train.sh
- Estimation: ${scriptsDir}/th_estimate.sh

When skills reference \${CLAUDE_PLUGIN_ROOT}/scripts/..., substitute the paths above.
</training-hub-plugin>`;
  };

  return {
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      if (!config.skills.paths.includes(skillsDir)) {
        config.skills.paths.push(skillsDir);
      }
    },

    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
