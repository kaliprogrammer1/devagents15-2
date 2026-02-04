import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(req: Request) {
  try {
    const { command, args = [], session = 'default' } = await req.json();
    
    // Safety check for commands
    const allowedCommands = ['open', 'snapshot', 'click', 'fill', 'scroll', 'close', 'get', 'metadata', 'screenshot', 'press', 'type'];
    if (!allowedCommands.includes(command)) {
      return NextResponse.json({ success: false, error: 'Invalid command' }, { status: 400 });
    }

    // Special handling for screenshot to return base64
    if (command === 'screenshot') {
      const tmpPath = `/tmp/screenshot-${session}-${Date.now()}.png`;
      await execAsync(`npx agent-browser screenshot ${tmpPath} --session=${session}`, { timeout: 30000 });
      const fs = require('fs');
      if (fs.existsSync(tmpPath)) {
        const screenshot = fs.readFileSync(tmpPath, { encoding: 'base64' });
        fs.unlinkSync(tmpPath);
        return NextResponse.json({ success: true, data: `data:image/png;base64,${screenshot}` });
      } else {
        return NextResponse.json({ success: false, error: 'Screenshot failed' });
      }
    }

    const cmdParts = ['npx agent-browser', command, ...args, `--session=${session}`];
    const fullCmd = cmdParts.join(' ');

    const { stdout, stderr } = await execAsync(fullCmd, { timeout: 30000 });
    
    if (stderr && !stderr.includes('debug') && !stderr.includes('warn')) {
      return NextResponse.json({ success: false, error: stderr }, { status: 500 });
    }

    return NextResponse.json({ success: true, data: stdout });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
