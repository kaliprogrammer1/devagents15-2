import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const SKILLS_FILE = path.join(process.cwd(), 'skills.md');

export async function GET() {
  try {
    if (!fs.existsSync(SKILLS_FILE)) {
      return NextResponse.json({ skills: [] });
    }

    const content = fs.readFileSync(SKILLS_FILE, 'utf-8');
    const skills = parseSkills(content);
    return NextResponse.json({ skills });
  } catch (error) {
    console.error('Error reading skills:', error);
    return NextResponse.json({ error: 'Failed to read skills' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const { action, skill } = await request.json();

    if (action === 'save') {
      let content = '';
      if (fs.existsSync(SKILLS_FILE)) {
        content = fs.readFileSync(SKILLS_FILE, 'utf-8');
      } else {
        content = '# Skills\n\n';
      }

      const skillMarkdown = `## ${skill.title}\n- **ID**: ${skill.id}\n- **Trigger**: ${skill.trigger}\n- **Actions**:\n${skill.actions.map((a: string) => `  - ${a}`).join('\n')}\n\n`;
      
      content += skillMarkdown;
      fs.writeFileSync(SKILLS_FILE, content);
      
      return NextResponse.json({ success: true });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (error) {
    console.error('Error saving skill:', error);
    return NextResponse.json({ error: 'Failed to save skill' }, { status: 500 });
  }
}

function parseSkills(content: string) {
  const skills = [];
  const sections = content.split('## ').slice(1);

  for (const section of sections) {
    const lines = section.split('\n');
    const title = lines[0].trim();
    const idMatch = section.match(/- \*\*ID\*\*: (.*)/);
    const triggerMatch = section.match(/- \*\*Trigger\*\*: (.*)/);
    
    const actionsStart = lines.findIndex(l => l.includes('- **Actions**:'));
    const actions = [];
    if (actionsStart !== -1) {
      for (let i = actionsStart + 1; i < lines.length; i++) {
        if (lines[i].startsWith('  - ')) {
          actions.push(lines[i].replace('  - ', '').trim());
        } else if (lines[i].trim() === '' || lines[i].startsWith('##')) {
          break;
        }
      }
    }

    skills.push({
      id: idMatch ? idMatch[1] : `skill-${Date.now()}`,
      title,
      trigger: triggerMatch ? triggerMatch[1] : '',
      actions,
      createdAt: Date.now(),
    });
  }

  return skills;
}
