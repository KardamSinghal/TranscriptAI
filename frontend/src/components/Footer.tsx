import styles from './Footer.module.css';

export function Footer() {
  return (
    <footer className={styles.footer}>
      <p className={styles.text}>Want to build AI apps like this?</p>
      <a
        href="https://www.skool.com/ai-engineer/about"
        target="_blank"
        rel="noopener noreferrer"
        className={styles.link}
      >
        Join the AI-Native Engineer community →
      </a>
    </footer>
  );
}
